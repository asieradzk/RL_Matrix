using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix.Memories
{
    public interface IMemory<TState>
    {
        void Push(Transition<TState> transition);
        List<Transition<TState>> Sample();
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
