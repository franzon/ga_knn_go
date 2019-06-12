package main

import (
	"github.com/bradfitz/slice"
)

type DistanceTag struct {
	distance float64
	tag      string
}

func Knn(trainingDataset *Dataset, testingDataset *Dataset, k int) float64 {

	correctResponses := 0.0

	for index := 0; index < len(testingDataset.Rows); index++ {
		distances := make([]DistanceTag, 0)

		for j := 0; j < len(trainingDataset.Rows); j++ {

			distance := testingDataset.Rows[index].EuclideanDistance(trainingDataset.Rows[j])
			distanceTag := DistanceTag{distance: distance, tag: trainingDataset.Rows[j].Tag}
			distances = append(distances, distanceTag)
		}

		slice.Sort(distances[:], func(i, j int) bool {
			return distances[i].distance < distances[j].distance
		})

		vote := distances[0].tag

		if vote == trainingDataset.Rows[index].Tag {
			correctResponses++
		}

	}

	return correctResponses / float64(len(testingDataset.Rows))
}
