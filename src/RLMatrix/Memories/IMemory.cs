using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix.Memories
{
    public interface IMemory<TState>
    {
        public void Push(Transition<TState> transition);
        public List<Transition<TState>> Sample(int sampleSize);
        public List<Transition<TState>> Sample();
        public void Save(string pathToFile);
        public void Load(string pathToFile);
        public int myCount { get; }
    }
    public interface IStorableMemory
    {
        void Save(string pathToFile);
        void Load(string pathToFile);
    }


    public interface IPERMemory<TState> : IMemory<TState>
    {
        // Overrides the Push method to add a priority.
        void Push(Transition<TState> transition, float priority);

        // Add an Update method specific to PER for updating priorities.
        void Update(long experienceId, float newPriority);
    }
    public interface IEpisodicMemory<TState> : IMemory<TState>
    {
        void Push(List<Transition<TState>> episode);
        public void ClearMemory();
    }
    public interface IBatchMemory<TState> : IMemory<TState>
    {
        void PushBatch(List<Transition<TState>> transitions);
    }



}
