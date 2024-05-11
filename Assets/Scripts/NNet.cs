using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using System;

using Random = UnityEngine.Random;

public class NNet : MonoBehaviour
{
    private Matrix<float> inputLayer = Matrix<float>.Build.Dense(1, 3);
    
    public List<Matrix<float>> hiddenLayers = new List<Matrix<float>>();

    private Matrix<float> outputLayer = Matrix<float>.Build.Dense(1, 2);

    public List<Matrix<float>> weights = new List<Matrix<float>>();

    public List<float> biases = new List<float>();

    public float fitness = 0f;

    public void Initialize(int hiddenLayerCount, int hiddenNeuronsCount)
    {
        //clear al list to erase data from previous run
        inputLayer.Clear();
        hiddenLayers.Clear();
        outputLayer.Clear();
        weights.Clear();
        biases.Clear();


        for (int i = 0; i < hiddenLayerCount; i++)
        {
            //add neurons for each hidden layer
            Matrix<float> f = Matrix<float>.Build.Dense(1, hiddenNeuronsCount);
            hiddenLayers.Add(f);


            biases.Add(Random.Range(-1f, 1f));

            //weights
            if(i == 0)
            {
                Matrix<float> inputToH1 = Matrix<float>.Build.Dense(3, hiddenNeuronsCount);
                weights.Add(inputToH1);
            }

            //list of connection in this layer beetwen neurons
            Matrix<float> hiddenToHidden = Matrix<float>.Build.Dense(hiddenNeuronsCount, hiddenNeuronsCount);
            weights.Add(hiddenToHidden);
        }

        //connections from last hidden layer to output layer
        Matrix<float> outputWeight = Matrix<float>.Build.Dense(hiddenNeuronsCount, 2);
        weights.Add(outputWeight);
        biases.Add(Random.Range(-1f, 1f));

        RandomiseWeights();
    }

    public NNet InitializeCopy(int hiddenLayerCount, int hiddenNeuronCount)
    {
        NNet n = new NNet();

        List<Matrix<float>> newWieghts = new List<Matrix<float>>();
        //n.weights = new List<Matrix<float>>();
        for(int i = 0; i < weights.Count; i++)
        {
            //n.weights.Add(weights[i].Clone());

            newWieghts.Add(weights[i].Clone());

            //Matrix<float> n = Matrix<float>.Build.Dense(weights[i].RowCount, weights[i].ColumnCount);

            //for (int x = 0; x < weights[i].RowCount; x++)
            //{
            //    for (int y = 0; y < weights[i].ColumnCount; y++)
            //    {
            //        n.weights[i][x, y] = weights[i][x, y];
            //    }
            //}
            
            //n.weights.Add(n);
        }

        List<float> newBiases = new List<float>();
        newBiases.AddRange(biases);

        n.weights = newWieghts;
        n.biases = newBiases;

        n.InitializeHidden(hiddenLayerCount, hiddenNeuronCount);

        return n;
    }

    public void InitializeHidden(int hiddenLayerCount, int hiddenNeuronCount)
    {
        inputLayer.Clear();
        hiddenLayers.Clear();
        outputLayer.Clear();

        for(int i = 0; i < hiddenLayerCount; i++)
        {
            Matrix<float> f = Matrix<float>.Build.Dense(1, hiddenNeuronCount);
            hiddenLayers.Add(f);
        }
    }

    private void RandomiseWeights()
    {
        for(int i = 0; i < weights.Count; i++)
        {
            for(int x = 0; x < weights[i].RowCount; x++)
            {
                for(int y = 0; y < weights[i].ColumnCount; y++)
                {
                    weights[i][x, y] = Random.Range(-1f, 1f);
                }
            }
        }
    }

    public (float, float) RunNetwork(float a, float b, float c)
    {
        inputLayer[0, 0] = a;
        inputLayer[0, 1] = b;
        inputLayer[0, 2] = c;

        inputLayer = inputLayer.PointwiseTanh();

        hiddenLayers[0] = ((inputLayer * weights[0]) + biases[0]).PointwiseTanh();

        for(int i = 1; i < hiddenLayers.Count; i++)
        {
            hiddenLayers[i] = ((hiddenLayers[i - 1] * weights[i]) + biases[i]).PointwiseTanh();
        }

        outputLayer = ((hiddenLayers[hiddenLayers.Count - 1] * weights[weights.Count - 1]) + biases[biases.Count - 1]).PointwiseTanh();

        //first output is acceleration and second is rotation
        return (Sigmoid(outputLayer[0, 0]), (float)Math.Tanh(outputLayer[0, 1]));
    }

    private float Sigmoid (float s)
    {
        return (1 / (1 + Mathf.Exp(-s)));
    }
}
