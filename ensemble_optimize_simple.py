import numpy as np
from tqdm import tqdm
import torch
from heapq import heappush, heappop, heappushpop
import math
import time
import matplotlib.pyplot as plotter

from scipy import optimize
from ela import generator

import utils
import metrics
import datasets

CAPACITY_INCREMENT = 1000


class _Simplex:
    def __init__(self, pointIndices, testCoords, contentFractions, objectiveScore, opportunityCost, contentFraction,
                 difference):
        self.pointIndices = pointIndices
        self.testCoords = testCoords
        self.contentFractions = contentFractions
        self.contentFraction = contentFraction
        self.__objectiveScore = objectiveScore
        self.__opportunityCost = opportunityCost
        self.update(difference)

    def update(self, difference):
        self.acquisitionValue = -(self.__objectiveScore + (self.__opportunityCost * difference))
        self.difference = difference

    def __eq__(self, other):
        return self.acquisitionValue == other.acquisitionValue

    def __lt__(self, other):
        return self.acquisitionValue < other.acquisitionValue


class SimpleTuner:
    def __init__(self, cornerPoints, objectiveFunction, exploration_preference=0.15):
        self.__cornerPoints = cornerPoints
        self.__numberOfVertices = len(cornerPoints)
        self.queue = []
        self.capacity = self.__numberOfVertices + CAPACITY_INCREMENT
        self.testPoints = np.empty((self.capacity, self.__numberOfVertices))
        self.objective = objectiveFunction
        self.iterations = 0
        self.maxValue = None
        self.minValue = None
        self.bestCoords = []
        self.opportunityCostFactor = exploration_preference  # / self.__numberOfVertices

    def optimize(self, maxSteps=10):
        for step in range(maxSteps):
            # print(self.maxValue, self.iterations, self.bestCoords)
            if len(self.queue) > 0:
                targetSimplex = self.__getNextSimplex()
                newPointIndex = self.__testCoords(targetSimplex.testCoords)
                for i in range(0, self.__numberOfVertices):
                    tempIndex = targetSimplex.pointIndices[i]
                    targetSimplex.pointIndices[i] = newPointIndex
                    newContentFraction = targetSimplex.contentFraction * targetSimplex.contentFractions[i]
                    newSimplex = self.__makeSimplex(targetSimplex.pointIndices, newContentFraction)
                    heappush(self.queue, newSimplex)
                    targetSimplex.pointIndices[i] = tempIndex
            else:
                testPoint = self.__cornerPoints[self.iterations]
                testPoint.append(0)
                testPoint = np.array(testPoint, dtype=np.float64)
                self.__testCoords(testPoint)
                if self.iterations == (self.__numberOfVertices - 1):
                    initialSimplex = self.__makeSimplex(np.arange(self.__numberOfVertices, dtype=np.intp), 1)
                    heappush(self.queue, initialSimplex)
            self.iterations += 1

    def get_best(self):
        return (self.maxValue, self.bestCoords[0:-1])

    def __getNextSimplex(self):
        targetSimplex = heappop(self.queue)
        currentDifference = self.maxValue - self.minValue
        while currentDifference > targetSimplex.difference:
            targetSimplex.update(currentDifference)
            # if greater than because heapq is in ascending order
            if targetSimplex.acquisitionValue > self.queue[0].acquisitionValue:
                targetSimplex = heappushpop(self.queue, targetSimplex)
        return targetSimplex

    def __testCoords(self, testCoords):
        objectiveValue = self.objective(testCoords[0:-1])
        if self.maxValue == None or objectiveValue > self.maxValue:
            self.maxValue = objectiveValue
            self.bestCoords = testCoords
            if self.minValue == None: self.minValue = objectiveValue
        elif objectiveValue < self.minValue:
            self.minValue = objectiveValue
        testCoords[-1] = objectiveValue
        if self.capacity == self.iterations:
            self.capacity += CAPACITY_INCREMENT
            self.testPoints.resize((self.capacity, self.__numberOfVertices))
        newPointIndex = self.iterations
        self.testPoints[newPointIndex] = testCoords
        return newPointIndex

    def __makeSimplex(self, pointIndices, contentFraction):
        vertexMatrix = self.testPoints[pointIndices]
        coordMatrix = vertexMatrix[:, 0:-1]
        barycenterLocation = np.sum(vertexMatrix, axis=0) / self.__numberOfVertices

        differences = coordMatrix - barycenterLocation[0:-1]
        distances = np.sqrt(np.sum(differences * differences, axis=1))
        totalDistance = np.sum(distances)
        barycentricTestCoords = distances / totalDistance

        euclideanTestCoords = vertexMatrix.T.dot(barycentricTestCoords)

        vertexValues = vertexMatrix[:, -1]

        testpointDifferences = coordMatrix - euclideanTestCoords[0:-1]
        testPointDistances = np.sqrt(np.sum(testpointDifferences * testpointDifferences, axis=1))

        inverseDistances = 1 / testPointDistances
        inverseSum = np.sum(inverseDistances)
        interpolatedValue = inverseDistances.dot(vertexValues) / inverseSum

        currentDifference = self.maxValue - self.minValue
        opportunityCost = self.opportunityCostFactor * math.log(contentFraction, self.__numberOfVertices)

        return _Simplex(pointIndices.copy(), euclideanTestCoords, barycentricTestCoords, interpolatedValue,
                        opportunityCost, contentFraction, currentDifference)

    def plot(self):
        if self.__numberOfVertices != 3: raise RuntimeError('Plotting only supported in 2D')
        matrix = self.testPoints[0:self.iterations, :]

        x = matrix[:, 0].flat
        y = matrix[:, 1].flat
        z = matrix[:, 2].flat

        coords = []
        acquisitions = []

        for triangle in self.queue:
            coords.append(triangle.pointIndices)
            acquisitions.append(-1 * triangle.acquisitionValue)

        plotter.figure()
        plotter.tricontourf(x, y, coords, z)
        plotter.triplot(x, y, coords, color='white', lw=0.5)
        plotter.colorbar()

        plotter.figure()
        plotter.tripcolor(x, y, coords, acquisitions)
        plotter.triplot(x, y, coords, color='white', lw=0.5)
        plotter.colorbar()

        plotter.show()


experiments = [
    'nopoolrefinenet_dpn92_dual_hypercolumn_poly_lr_aux_data_pseudo_labels',
    'nopoolrefinenet_seresnext101_ndadam_scse_block_pseudo_labels'
]
output = False

test_predictions_experiment = []

for name in experiments:
    test_predictions = utils.TestPredictions('{}'.format(name), mode='val')
    test_predictions_experiment.append(test_predictions.load_raw())

train_samples = utils.get_train_samples()


transforms = generator.TransformationsGenerator([])
dataset = datasets.AnalysisDataset(train_samples, './data/train', transforms, utils.TestPredictions('{}'.format(name), mode='val').load())


def run_evaluation(weights):
    weights = weights[:2]
    weights_sum = np.sum(weights)
    split_map = []
    val = utils.get_train_samples()
    predictions = []
    masks = []

    for id in tqdm(val):
        _, mask, _ = dataset.get_by_id(id)
        prediction = torch.stack([torch.mean(torch.sigmoid(torch.FloatTensor(predictions[id])), dim=0) for predictions in test_predictions_experiment], dim=0)
        mask = torch.FloatTensor(mask)

        predictions.append(prediction)
        masks.append(mask)

    predictions = torch.stack(predictions, dim=0).cuda()
    masks = torch.stack(masks, dim=0).cuda()

    predictions = predictions * (torch.FloatTensor(weights) / float(weights_sum)).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(2).expand_as(predictions)
    predictions = torch.sum(predictions, dim=1)

    if output:
        ensemble_predictions = utils.TestPredictions(output, mode='val')
        ensemble_predictions.add_predictions(zip(predictions.cpu().numpy(), train_samples))
        ensemble_predictions.save()

    predictions = (predictions > 0.5).float()

    map = metrics.mAP(predictions, masks)
    split_map.append(map)

    return np.mean(split_map)


#mAP_mean = run_evaluation([1]*len(experiments))
#print('Uniform weight mAP: ', mAP_mean)

optimization_domain_vertices = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
number_of_iterations = 30
exploration = 0.15 # optional, default 0.15

# Optimize weights
tuner = SimpleTuner(optimization_domain_vertices, run_evaluation, exploration_preference=exploration)
tuner.optimize(number_of_iterations)
best_objective_value, best_weights = tuner.get_best()

print('Best objective value =', best_objective_value)
print('Optimum weights =', best_weights)
print('Ensembled Accuracy (same as best objective value) =', run_evaluation(best_weights))