package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"github.com/bradfitz/slice"
	"github.com/logrusorgru/aurora"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Individual Representa um indivíuo
type Individual struct {
	Genome  []byte
	Fitness float64
}

// DatasetRow Representa uma entrada em um Dataset
type DatasetRow struct {
	Features []float64
	Tag      string
}

// Dataset Representa um Dataset
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

// ComputeFitness Calcula o fitness usando knn
func ComputeFitness(population []Individual, trainingDataset *Dataset, testingDataset *Dataset) Individual {

	var wg sync.WaitGroup

	wg.Add(len(population))

	for index := 0; index < len(population); index++ {

		go func(i int) {

			population[i].Fitness = Knn(population[i], trainingDataset, testingDataset)
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

// SelectParents Seleciona os pais
func SelectParents(population []Individual) []Individual {
	parents := make([]Individual, 0)

	for index := 0; index < int(math.Ceil(float64(len(population))/2)); index++ {
		parents = append(parents, population[index])
	}

	return parents
}

// Crossover Realiza o cruzamento
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

// Mutate Realiza mutação
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

// LoadDataset Carrega um Dataset
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
	mutationRate := 0.01
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
		localFitness := ComputeFitness(population, trainingDataset, testingDataset)
		fmt.Println("Fitness: ", localFitness.Fitness)

		if localFitness.Fitness > bestIndividual.Fitness {
			bestIndividual = localFitness
			fmt.Println(aurora.Green("Novo fitness máximo: "), aurora.Bold(aurora.Green(bestIndividual.Fitness)))

			tmp := make([]string, nFeatures)
			for z := 0; z < nFeatures; z++ {
				tmp[z] = strconv.Itoa(int(bestIndividual.Genome[z]))
			}
			fmt.Println(aurora.Blue("Melhor genoma: "), aurora.Bold(aurora.Blue(strings.Join(tmp, ","))))

		}
		parents := SelectParents(population)
		childs := Crossover(parents)
		mutated := Mutate(childs, mutationRate)

		population = mutated[:]

		fmt.Println("Tempo para executar iteração (segundos): ", time.Now().Sub(now).Seconds())
		fmt.Println("")
	}

	fmt.Println(aurora.Red("Execução finalizada."))
	fmt.Println(aurora.Green("Fitness máximo obtido: "), aurora.Bold(aurora.Green(bestIndividual.Fitness)))
	fmt.Println(aurora.Blue("Melhor genoma: "), aurora.Bold(aurora.Blue(bestIndividual.Genome)))
	fmt.Println("Tempo total (segundos): ", time.Now().Sub(first).Seconds())

}
