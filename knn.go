package main

import (
	 "github.com/bradfitz/slice"

	 "github.com/umpc/go-sortedmap"
	 "github.com/umpc/go-sortedmap/asc"
)

type DistanceTag struct {
	distance float64
	tag string
}

func Knn(trainingDataset Dataset, testingDataset Dataset, k int) float64 {

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


		distances = distances[:k]

		sm := sortedmap.New(k,  asc.Int)

		for index := 0; index < k; index++ {
			if !sm.Has(index) {
				sm.Insert(distances[index].tag, 0)
			}

			val, ok := sm.Get(distances[index].tag); 
			if ok {
				sm.Replace(distances[index].tag, val.(int)+1)
			}
		}

		vote := sm.Keys()[len(sm.Keys())-1]
		// populationSize := 10
		// nFeatures := 7

		if vote == trainingDataset.Rows[index].Tag {
			correctResponses++
		}

	}

	return correctResponses / float64(len(testingDataset.Rows))
}