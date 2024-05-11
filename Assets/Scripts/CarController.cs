using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(NNet))]
public class CarController : MonoBehaviour
{
    private Vector3 startPosition, startRotation;

    private NNet network;

    [Range(-1f, 1f)]
    public float a, t;

    public float timeSinceStart = 0f;

    [Header("Fitness")]
    public float overallFitness = 0f;

    //this two variables are used to calculate how important in fitness are this two results
    public float distanceMultiplier = 1.4f;
    public float avgSpeedMultiplier = 0.2f;
    public float sensorMultiplier = 0.1f;

    //results of this trial
    private Vector3 lastPosition;
    private float totalDistanceTravelled = 0f;
    private float avgSpeed = 0f;

    [Header("NetworkOptions")]
    public int Layers = 1;
    public int Neurons = 10;

    //distance from canr to the walls
    private float aSensor, bSensor, cSensor;

    Transform _transform;

    private void Awake()
    {
        _transform = transform;
        startPosition = _transform.position;
        startRotation = _transform.eulerAngles;
        network = GetComponent<NNet>();


        ////TEST CODE
        //network.Initialize(Layers, Neurons);
    }

    public void ResetWithNetwork(NNet net)
    {
        network = net;
        Reset();
    }


    public void Reset()
    {
        ////TEST CODE
        //network.Initialize(Layers, Neurons);

        timeSinceStart = 0f;
        totalDistanceTravelled = 0f;
        avgSpeed = 0f;
        lastPosition = startPosition;
        overallFitness = 0f;
        _transform.position = startPosition;
        _transform.eulerAngles = startRotation;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.tag == "Wall")
        {
            Die();
        }
    }

    public void Die()
    {
        GeneticManager.instance.Death(overallFitness, network);
    }

    private void FixedUpdate()
    {
        InputSensers();
        timeSinceStart += Time.deltaTime;

        lastPosition = _transform.position;

        (a, t) = network.RunNetwork(aSensor, bSensor, cSensor);

        MoveCar(a, t);

        CalcFittnes();
    }

    private void CalcFittnes()
    {
        totalDistanceTravelled += Vector3.Distance(_transform.position, lastPosition);
        avgSpeed = totalDistanceTravelled / timeSinceStart;

        overallFitness = (totalDistanceTravelled * distanceMultiplier) + (avgSpeed * avgSpeedMultiplier) + ((aSensor + bSensor + cSensor) * sensorMultiplier);
    
        if(timeSinceStart > 20 && overallFitness < 40)
        {
            Die();
        }

        if(overallFitness >= 1000)
        {
            //save neural network to json
            Die();
        }
    }

    private void InputSensers()
    {
        Vector3 a = _transform.forward + transform.right;
        Vector3 b = _transform.forward;
        Vector3 c = _transform.forward - transform.right;

        Ray r = new Ray(_transform.position, a);
        RaycastHit hit;

        if(Physics.Raycast(r, out hit))
        {
            aSensor = hit.distance / 20;//division by 20 is to normalize the value
            Debug.DrawLine(r.origin, hit.point, Color.red);
            //Debug.Log("aSensor: " + aSensor);
        }

        r.direction = b;
        if (Physics.Raycast(r, out hit))
        {
            bSensor = hit.distance / 20;//division by 20 is to normalize the value
            Debug.DrawLine(r.origin, hit.point, Color.red);
            //Debug.Log("bSensor: " + bSensor);
        }

        r.direction = c;
        if (Physics.Raycast(r, out hit))
        {
            cSensor = hit.distance / 20;//division by 20 is to normalize the value
            Debug.DrawLine(r.origin, hit.point, Color.red);
            //Debug.Log("cSensor: " + cSensor);
        }
    }


    private Vector3 input;
    public void MoveCar(float v, float h)
    {
        //position
        input = Vector3.Lerp(Vector3.zero, new Vector3(0, 0, v * 11.4f), 0.02f);
        input = _transform.TransformDirection(input);
        transform.position += input;

        //rotation
        transform.eulerAngles += new Vector3(0, h * 90*0.02f, 0);
    }
}
