using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using RLMatrix.Agents.Common;

namespace RLMatrix.Memories
{
    public interface IMemory<TState>
    {
        int Length { get; }
        public int NumEpisodes { get;}
        void Push(TransitionInMemory<TState> transition);
        void Push (IList<TransitionInMemory<TState>> transitions);
        IList<TransitionInMemory<TState>> SampleEntireMemory();
        IList<TransitionInMemory<TState>> Sample(int batchSize); 
        void ClearMemory();
    }

    public interface IStorableMemory
    {
        void Save(string pathToFile);
        void Load(string pathToFile);
    }

    public interface IPERMemory<TState> : IMemory<TState>
    {
        void Push(TransitionInMemory<TState> transition, float priority);
        void Push(IEnumerable<TransitionInMemory<TState>> transitions, IEnumerable<float> priorities);
        void Update(int experienceId, float newPriority);
    }

    public interface IEpisodicMemory<TState> : IMemory<TState>
    {
        void Push(List<TransitionInMemory<TState>> episode);
        void ClearMemory();
    }
}