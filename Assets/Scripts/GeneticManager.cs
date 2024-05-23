using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

public class GeneticManager: MonoBehaviour
{
    public static GeneticManager instance;

    [Header("references")]
    public CarController controller;

    [Header("Controls")]
    public int initialPopulation = 85;
    [Range(0.0f, 1.0f)]
    public float mutationRate = 0.055f;

    [Header("Crossover Controls")]
    public int bestAgentSelection = 8;
    public int worstAgentSelection = 3;
    public int numberToCrossover;

    private List<int> genPool = new List<int>();

    private int naturallySelected;

    private NNet[] population;

    [Header("Public View")]
    public int currentGeneration;
    public int currentGenome = 0;

    private void Start()
    {
        instance = this;
        CreatePopulation();
    }

    private void CreatePopulation()
    {
        population = new NNet[initialPopulation];
        FillPopulationWithRandomValues(population, 0);
        ResetToCurrentGenome();
    }

    private void FillPopulationWithRandomValues(NNet[] newPopulation, int startingIndex)
    {
        while(startingIndex < initialPopulation)
        {
            newPopulation[startingIndex] = new NNet();
            newPopulation[startingIndex].Initialize(controller.Layers, controller.Neurons);
            startingIndex++;
        }
    }

    private void ResetToCurrentGenome()
    {
        controller.ResetWithNetwork(population[currentGenome]);
    }

    public void Death(float fittest, NNet network)
    {
        if(currentGenome < population.Length - 1)
        {
            population[currentGenome].fitness = fittest;
            currentGenome++;
            ResetToCurrentGenome();
        }
        else
        {
            Repopulate();
        }
    }

    private void Repopulate()
    {
        genPool.Clear();
        currentGeneration++;
        naturallySelected = 0;

        SortPopulation();

        NNet[] newPopulation = PickBestPopulation();

        Crossover(newPopulation);
        Mutate(newPopulation);

        FillPopulationWithRandomValues(newPopulation, naturallySelected);

        population = newPopulation;

        currentGenome = 0;

        ResetToCurrentGenome();
    }

    private void Crossover(NNet[] newPopulation)
    {
        for(int i = 0; i < numberToCrossover; i+=2)
        {
            int aIndex = i;
            int bIndex = i + 1;

            if(genPool.Count >= 1)
            {
                for(int j = 0; j<100; j++)
                {
                    aIndex = genPool[Random.Range(0, genPool.Count)];
                    bIndex = genPool[Random.Range(0, genPool.Count)];

                    if(aIndex != bIndex)
                    {
                        break;
                    }
                }
            }

            NNet child1 = new NNet();
            NNet child2 = new NNet();

            child1.Initialize(controller.Layers, controller.Neurons);
            child2.Initialize(controller.Layers, controller.Neurons);

            child1.fitness = 0;
            child2.fitness = 0;

            for(int j = 0; j < child1.weights.Count; j++)
            {
                if(Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    Debug.Log("child1 weights: " + child1.weights[j]);
                    Debug.Log("newPopulation[aIndex]" + newPopulation[aIndex]);
                    Debug.Log("newPopulation[aIndex].weights[j]: " + newPopulation[aIndex].weights[j]);
                    child1.weights[j] = newPopulation[aIndex].weights[j];
                    Debug.Log("child2 weights: " + child2.weights[j]);
                    Debug.Log("newPopulation[bIndex]" + newPopulation[bIndex]);
                    Debug.Log("newPopulation[bIndex].weights[j]: " + newPopulation[bIndex].weights[j]);
                    child2.weights[j] = newPopulation[bIndex].weights[j];
                }
                else
                {
                    Debug.Log("child1 weights: " + child1.weights[j]);
                    Debug.Log("newPopulation[aIndex]" + newPopulation[bIndex]);
                    Debug.Log("newPopulation[aIndex].weights[j]: " + newPopulation[bIndex].weights[j]);
                    child1.weights[j] = newPopulation[bIndex].weights[j];
                    Debug.Log("child2 weights: " + child2.weights[j]);
                    Debug.Log("newPopulation[bIndex]" + newPopulation[aIndex]);
                    Debug.Log("newPopulation[bIndex].weights[j]: " + newPopulation[aIndex].weights[j]);
                    child2.weights[j] = newPopulation[aIndex].weights[j];
                }
            }

            for (int j = 0; j < child1.biases.Count; j++)
            {
                if (Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    child1.biases[j] = newPopulation[aIndex].biases[j];
                    child2.biases[j] = newPopulation[bIndex].biases[j];
                }
                else
                {
                    child1.biases[j] = newPopulation[bIndex].biases[j];
                    child2.biases[j] = newPopulation[aIndex].biases[j];
                }
            }

            newPopulation[naturallySelected] = child1;
            naturallySelected++;

            newPopulation[naturallySelected] = child2;
            naturallySelected++;
        }
    }

    private void Mutate(NNet[] newPopulation)
    {
        for(int i = 0; i<naturallySelected; i++)
        {
            for(int j = 0; j < newPopulation[i].weights.Count; j++)
            {
                for(int k = 0; k < newPopulation[i].weights[j].RowCount; k++)
                {
                    if(Random.Range(0.0f, 1.0f) < mutationRate)
                    {
                        newPopulation[i].weights[j] = MutateMatrix(newPopulation[i].weights[j]);
                    }
                }
            }
        }
    }

    Matrix<float> MutateMatrix(Matrix<float> a)
    {
        int randomPoints = Random.Range(1, (a.RowCount * a.ColumnCount) / 7);

        Matrix<float> temp = a;

        for(int i = 0; i < randomPoints; i++)
        {
            int randomColumn = Random.Range(0, a.ColumnCount);
            int randomRow = Random.Range(0, a.RowCount);

            temp[randomRow, randomColumn] = Mathf.Clamp(temp[randomRow, randomColumn] + Random.Range(-1.0f, 1.0f), -1f, 1f);
        }

        return temp;
    }

    public NNet[] PickBestPopulation()
    {
        NNet[] newPopulation = new NNet[initialPopulation];

        for(int i = 0; i < bestAgentSelection; i++)
        {
            newPopulation[naturallySelected] = population[i].InitializeCopy(controller.Layers, controller.Neurons);
            newPopulation[naturallySelected].fitness = 0;

            naturallySelected++;

            int f = Mathf.RoundToInt(population[i].fitness * 10);

            for(int j = 0; j < f; j++)
            {
                genPool.Add(i);
            }
        }

        for(int i = 0; i < worstAgentSelection; i++)
        {
            int last = population.Length - 1;
            last -= i;

            int f = Mathf.RoundToInt(population[last].fitness * 10);
            for(int j = 0; j < f; j++)
            {
                genPool.Add(last);
            }
        }

        return newPopulation;
    }

    private void SortPopulation()
    {
        //sort by fitness using bubble sort
        for(int i = 0; i < population.Length; i++)
        {
            for(int j = i; j < population.Length; j++)
            {
                if (population[i].fitness < population[j].fitness)
                {
                    NNet temp = population[i];
                    population[i] = population[j];
                    population[j] = temp;
                }
            }
        }
    }
}
