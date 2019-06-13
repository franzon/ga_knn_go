package main

import (
	"math"
)

// DistanceTag Distância e tag
type DistanceTag struct {
	distance float64
	tag      string
}

// EuclideanDistance Distância euclidiana
func EuclideanDistance(bitString []byte, row1, row2 *DatasetRow) float64 {
	distance := 0.0

	for i := 0; i < len(bitString); i++ {
		if bitString[i] == 1 {
			distance += math.Pow(row2.Features[i]-row1.Features[i], 2)
		}
	}

	return math.Sqrt(distance)
}

// Knn Função k-NN
func Knn(individual Individual, trainingDataset *Dataset, testingDataset *Dataset) float64 {
	correctResponses := 0.0

	for i := 0; i < len(testingDataset.Rows); i++ {

		nearest := DistanceTag{distance: math.MaxFloat64}
		for j := 0; j < len(trainingDataset.Rows); j++ {

			distance :=  EuclideanDistance(individual.Genome, &testingDataset.Rows[i], &trainingDataset.Rows[j])
			if distance < nearest.distance {
				nearest = DistanceTag{distance: distance, tag: trainingDataset.Rows[j].Tag}
			}
		}

		if testingDataset.Rows[i].Tag == nearest.tag {
			correctResponses++
		}
	}

	return correctResponses / float64(len(testingDataset.Rows))
}
