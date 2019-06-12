package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"github.com/bradfitz/slice"
	. "github.com/logrusorgru/aurora"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
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
		distance += math.Pow(y.Features[index]-x.Features[index], 2)
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

func ComputeFitness(population []Individual, trainingDataset *Dataset, testingDataset *Dataset, k int) Individual {

	var wg sync.WaitGroup

	wg.Add(len(population))

	for index := 0; index < len(population); index++ {

		go func(i int) {
			preparedTrainingDataset := GetPreparedDataset(population[i], trainingDataset)
			preparedTestingDataset := GetPreparedDataset(population[i], testingDataset)
			population[i].Fitness = Knn(&preparedTrainingDataset, &preparedTestingDataset, k)
			defer wg.Done()
		}(index)
	}

	wg.Wait()

	// Ordena do maior fitness para o menor
	slice.Sort(population[:], func(i, j int) bool {
		return population[i].Fitness > population[j].Fitness
	})

	return population[0]
}

// SelectParents
func SelectParents(population []Individual) []Individual {
	parents := make([]Individual, 0)

	for index := 0; index < len(population)/2; index++ {
		parents = append(parents, population[index])
	}

	return parents
}

func Crossover(population []Individual) []Individual {

	childs := make([]Individual, 0)

	for index := 0; index < len(population); index++ {

		idx1 := rand.Intn(len(population))
		idx2 := rand.Intn(len(population))

		splitPoint := rand.Intn(len(population))

		child1 := Individual{Fitness: 0.0, Genome: make([]byte, 0)}
		child1.Genome = append(population[idx1].Genome[:splitPoint+1], population[idx2].Genome[splitPoint+1:]...)

		child2 := Individual{Fitness: 0.0, Genome: make([]byte, 0)}
		child2.Genome = append(population[idx2].Genome[:splitPoint+1], population[idx1].Genome[splitPoint+1:]...)

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

	first := time.Now()

	fmt.Println("Algoritmo Genético + k-NN")
	fmt.Println("Cores CPU: ", runtime.NumCPU())

	populationSize := 100
	nFeatures := 132
	k := 1
	mutationRate := 0.05
	iterations := 500

	bestIndividual := Individual{Fitness: 0.0}

	population := InitPopulation(populationSize, nFeatures)

	fmt.Println("Carregando datasets...")
	trainingDataset, _ := LoadDataset("./treinamento.txt")
	testingDataset, _ := LoadDataset("./teste.txt")

	fmt.Println("Iniciando execução")

	for index := 0; index < iterations; index++ {

		fmt.Println("Iteração: ", index)

		now := time.Now()
		localFitness := ComputeFitness(population, trainingDataset, testingDataset, k)
		fmt.Println("Fitness: ", localFitness.Fitness)

		parents := SelectParents(population)
		childs := Crossover(parents)
		mutated := Mutate(childs, mutationRate)

		if localFitness.Fitness > bestIndividual.Fitness {
			bestIndividual = localFitness
			fmt.Println(Green("Novo fitness máximo: "), Bold(Green(bestIndividual.Fitness)))
			fmt.Println(Blue("Melhor genoma: "), Bold(Blue(bestIndividual.Genome)))
		}

		population = mutated[:]

		fmt.Println("Tempo para executar iteração (segundos): ", time.Now().Sub(now).Seconds())
		fmt.Println("")
	}

	fmt.Println(Red("Execução finalizada."))
	fmt.Println(Green("Fitness máximo obtido: "), Bold(Green(bestIndividual.Fitness)))
	fmt.Println(Blue("Melhor genoma: "), Bold(Blue(bestIndividual.Genome)))
	fmt.Println("Tempo total (segundos): ", time.Now().Sub(first).Seconds())

}
