package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

type Individual struct {
	Genome  []byte
	Fitness float64
}

type DatasetRow struct {
	Features []float64
	Tag      string
}
type Dataset struct {
	Rows []DatasetRow
}

// InitPopulation Inicia população
func InitPopulation(size int, nFeatures int) []Individual {

	population := make([]Individual, 0)
	for index := 0; index < size; index++ {
		individual := Individual{Genome: make([]byte, 0), Fitness: 0.0}

		for j := 0; j < nFeatures; j++ {
			rnd := byte(rand.Intn(2))
			individual.Genome = append(individual.Genome, rnd)
		}

		population = append(population, individual)
	}

	return population
}

func GetPreparedDataset(individual Individual, dataset *Dataset) Dataset {
	newDataset := Dataset{Rows: make([]DatasetRow, 0)}

	// for index := 0; index < len(individual.Genome); index++ {
	// 	if individual.Genome[index] == 1 {
	// 		newDataset.Rows = append(newDataset.Rows, dataset.Rows[index])
	// 	}
	// }

	for i := 0; i < len(dataset.Rows); i++ {

		row := DatasetRow{Features: make([]float64, 0), Tag: dataset.Rows[i].Tag}

		for j := 0; j < len(dataset.Rows[i].Features); j++ {

			if individual.Genome[j] == 1 {
				row.Features = append(row.Features, dataset.Rows[i].Features[j])
			}
		}

		newDataset.Rows = append(newDataset.Rows, row)
	}
	return newDataset
}

func ComputeFitness(population []Individual, trainingDataset *Dataset, testingDataset *Dataset) {

	for index := 0; index < len(population); index++ {
		preparedTrainingDataset := GetPreparedDataset(population[index], trainingDataset)
		preparedTestingDataset := GetPreparedDataset(population[index], testingDataset)

		// executa knn
		// atribui fitness

		fmt.Print(preparedTestingDataset, preparedTrainingDataset)

		population[index].Fitness = 0.7
	}
}

func LoadDataset(filePath string) (*Dataset, error) {
	f, err := os.Open(filePath)
	defer f.Close()

	if err != nil {

		return nil, errors.New("error")
	}

	dataset := Dataset{Rows: make([]DatasetRow, 0)}
	lines, err := csv.NewReader(f).ReadAll()

	for _, line := range lines {
		row := DatasetRow{Features: make([]float64, 0)}

		for index := 0; index < len(line)-1; index++ {
			f, err := strconv.ParseFloat(line[index], 64)
			if err != nil {
				return nil, errors.New("error")
			}
			row.Features = append(row.Features, f)
		}

		row.Tag = line[len(line)-1]

		dataset.Rows = append(dataset.Rows, row)
	}
	return &dataset, nil
}

func main() {

	// reader := csv.NewReader(bufio.NewReader(trainingFile))

	populationSize := 10
	nFeatures := 7

	population := InitPopulation(populationSize, nFeatures)

	trainingDataset, _ := LoadDataset("./treinamento-basic.txt")
	testingDataset, _ := LoadDataset("./treinamento-basic.txt")

	ComputeFitness(population, trainingDataset, testingDataset)
}
