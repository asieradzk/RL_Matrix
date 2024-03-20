using RLMatrix;
using RLMatrix.Memories;
using System.Runtime.Serialization.Formatters.Binary;

public class PrioritizedReplayMemory<TState> : IMemory<TState>, IStorableMemory
{
    private readonly int capacity;
    private readonly int batchSize;
    private readonly Random random = new Random();
    private readonly SumTree sumTree;
    private List<Transition<TState>> memory;
    private int currentIndex;
    private List<int> lastSampledIndices;

    public int Length => memory.Count;

    public int myCount => throw new NotImplementedException();

    public PrioritizedReplayMemory(int capacity, int batchSize)
    {
        this.capacity = capacity;
        this.batchSize = batchSize;
        this.sumTree = new SumTree(capacity);
        this.memory = new List<Transition<TState>>(capacity);
        this.currentIndex = 0;
        this.lastSampledIndices = new List<int>();
    }

    public void Push(Transition<TState> transition)
    {
        float priority = sumTree.MaxPriority;
        if (memory.Count < capacity)
        {
            memory.Add(transition);
        }
        else
        {
            //memory[currentIndex] = transition;
            memory.RemoveAt(0);
            memory.Add(transition);
        }
        sumTree.Add(priority);
        currentIndex = (currentIndex + 1) % capacity;
    }

    public List<Transition<TState>> Sample()
    {
        if (batchSize > Length)
        {
            throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
        }

        lastSampledIndices.Clear();
        List<Transition<TState>> sampledTransitions = new List<Transition<TState>>(batchSize);

        float segment = sumTree.TotalSum / batchSize;
        for (int i = 0; i < batchSize; i++)
        {
            float a = segment * i;
            float b = segment * (i + 1);
            float value = (float)(random.NextDouble() * (b - a) + a);
            int index = sumTree.Retrieve(value);
            sampledTransitions.Add(memory[index]);
            lastSampledIndices.Add(index);
        }

        return sampledTransitions;
    }

    public List<int> GetSampledIndices()
    {
        return new List<int>(lastSampledIndices);
    }

    public void UpdatePriority(int index, float priority)
    {
        if (index >= 0 && index < Length)
        {
            sumTree.Update(index, priority);
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range for priority update.");
        }
    }

    public void ClearMemory()
    {
        memory.Clear();
        sumTree.Clear();
        currentIndex = 0;
    }



        public void Save(string pathToFile)
        {
            using var fs = new FileStream(pathToFile, FileMode.Create);
            var bf = new BinaryFormatter();
            bf.Serialize(fs, memory);
        }

        public void Load(string pathToFile)
        {
            if (!File.Exists(pathToFile))
                throw new FileNotFoundException($"File {pathToFile} does not exist.");

            using var fs = new FileStream(pathToFile, FileMode.Open);
            var bf = new BinaryFormatter();
            memory = (List<Transition<TState>>)bf.Deserialize(fs);
            // Note: After loading, you might need to rebuild your SumTree based on loaded transitions
        }

    public List<Transition<TState>> Sample(int sampleSize)
    {
        throw new NotImplementedException();
    }
}

