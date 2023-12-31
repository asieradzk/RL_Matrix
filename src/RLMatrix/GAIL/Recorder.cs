﻿using RLMatrix.Memories;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix
{
    

    public class Recorder<T> 
    {
        public IMemory<T> myMemory;

        public Recorder()
        {
            myMemory = new TransitionReplayMemory<T>(10000, 512);
        }

        Transition<T>? previousTransition;
        public void AddStep(T observation, int[] discreteActions, float[] continousActions, float rewards)
        {
            var newTransition = new Transition<T>(DeepCopy(observation), discreteActions, continousActions, rewards, default);
           
            if(previousTransition != null)
            {
                var transition = new Transition<T>(previousTransition.state, previousTransition.discreteActions, 
                    previousTransition.continuousActions, previousTransition.reward, newTransition.state);
                myMemory.Push(transition);
            }
            previousTransition = newTransition;
        }

        public void EndEpisode()
        {
            if(previousTransition != null)
            {
                myMemory.Push(previousTransition);
            }
            previousTransition = null;
        }

        public void Save(string pathToFile)
        {
            myMemory.Save(pathToFile);
        }

        public void Load(string pathToFile)
        {
            myMemory.Load(pathToFile);
        }

        public T DeepCopy(T input)
        {
            if (!typeof(T).IsArray)
            {
                throw new InvalidOperationException("This method can only be used with arrays!");
            }

            // Handle nulls
            if (ReferenceEquals(input, null))
            {
                return default(T);
            }

            var rank = ((Array)(object)input).Rank;
            var lengths = new int[rank];
            for (int i = 0; i < rank; ++i)
                lengths[i] = ((Array)(object)input).GetLength(i);

            var clone = Array.CreateInstance(typeof(T).GetElementType(), lengths);

            Array.Copy((Array)(object)input, clone, ((Array)(object)input).Length);

            return (T)(object)clone;
        }

    }
}
