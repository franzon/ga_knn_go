package main

import (
	"sync"
	"github.com/bradfitz/slice"
	"runtime"
	"fmt"
	"encoding/csv"
	"errors"
	"math/rand"
	"os"
	"strconv"
	"math"
)

type Individual struct {
	Genome  []byte
	Fitness float64
}

type DatasetRow struct {
	Features []float64
	Tag      string
}
 
func (x *DatasetRow) EuclideanDistance(y DatasetRow) float64 {

	distance := 0.0

	for index := 0; index < len(x.Features); index++ {
		distance += math.Pow(y.Features[index] - x.Features[index], 2)
	}

	return math.Sqrt(distance)
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

		for j := 0; j < len(individual.Genome); j++ {

			if individual.Genome[j] == 1 {
				row.Features = append(row.Features, dataset.Rows[i].Features[j])
			}
		}

		newDataset.Rows = append(newDataset.Rows, row)
	}
	return newDataset
}

func ComputeFitness(population []Individual, trainingDataset *Dataset, testingDataset *Dataset, k int) float64 {


    var wg sync.WaitGroup

	wg.Add(len(population))


	for index := 0; index < len(population); index++ {
		// preparedTrainingDataset := GetPreparedDataset(population[index], trainingDataset)
		// preparedTestingDataset := GetPreparedDataset(population[index], testingDataset)

		// population[index].Fitness = Knn(preparedTrainingDataset, preparedTestingDataset, k)

		go func(i int) {
			preparedTrainingDataset := GetPreparedDataset(population[i], trainingDataset)
			preparedTestingDataset := GetPreparedDataset(population[i], testingDataset)
			population[i].Fitness = Knn(preparedTrainingDataset, preparedTestingDataset, k)
			defer wg.Done()		}(index)
	}

    wg.Wait()

	// Ordena do maior fitness para o menor
	slice.Sort(population[:], func(i, j int) bool {
		return population[i].Fitness > population[j].Fitness
	})

	return population[0].Fitness
}


// SelectParents 
func SelectParents(population []Individual) []Individual {
	parents := make([]Individual, 0)

	for index := 0; index < len(population) / 2; index++ {
		parents = append(parents, population[index])
	}

	return parents
}

func Crossover(population []Individual) []Individual {

	childs := make([]Individual, 0)

	for index := 0; index < len(population); index++ {
		
		idx1 := rand.Intn(len(population))
		idx2 := rand.Intn(len(population))

		middle := len(population)/2

		child1 := Individual{Fitness: 0.0, Genome: make([]byte, 0)}
		child1.Genome = append(population[idx1].Genome[:middle+1], population[idx2].Genome[middle+1:]...)

		child2 := Individual{Fitness: 0.0, Genome: make([]byte, 0)}
		child2.Genome = append(population[idx2].Genome[:middle+1], population[idx1].Genome[middle+1:]...)

		childs = append(childs, child1, child2)
	}

	return childs
}

func Mutate(population []Individual, mutationRate float64) []Individual {
	mutatedPopulation := make([]Individual, 0)

	for index := 0; index < len(population); index++ {
		mutatedIndividual := Individual{Genome: make([]byte, 0), Fitness: 0.0}

		for j := 0; j < len(population[index].Genome); j++ {

			bit := population[index].Genome[j]
			
			if rand.Float64() < mutationRate {
				if bit == 0 {
					bit = 1
				} else {
					bit = 0
				}
			}

			mutatedIndividual.Genome = append(mutatedIndividual.Genome, bit)
		}

		mutatedPopulation = append(mutatedPopulation, mutatedIndividual)
	}

	return mutatedPopulation
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

	runtime.GOMAXPROCS(12)


	// reader := csv.NewReader(bufio.NewReader(trainingFile))

	populationSize := 100
	nFeatures := 132
	k := 3
	mutationRate := 0.01
	iterations := 50

	maxFitness := 0.0

	population := InitPopulation(populationSize, nFeatures)

	trainingDataset, _ := LoadDataset("./treinamento.txt")
	testingDataset, _ := LoadDataset("./teste.txt")


	for index := 0; index < iterations; index++ {
		
		fmt.Println("Iteração ", index)

		localFitness  := ComputeFitness(population, trainingDataset, testingDataset, k)
		fmt.Println("Fitness ", localFitness)
	
		if (localFitness > maxFitness) {
			maxFitness = localFitness
			fmt.Println("Novo fitness máximo ", maxFitness)
		}
	
		parents := SelectParents(population)
		childs := Crossover(parents)
		mutated := Mutate(childs, mutationRate)
	
		population = mutated[:]
	}

}
