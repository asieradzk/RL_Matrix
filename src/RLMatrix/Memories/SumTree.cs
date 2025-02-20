namespace RLMatrix;

public class SumTree(int capacity)
{
    private readonly float[] _tree = new float[2 * capacity];
    private int _count;

    public float MaxPriority { get; private set; } = 1f;
    
    public float TotalSum => _tree[1]; // Total priority sum is at the root

    public void Update(int index, float priority)
    {
        var treeIndex = index + capacity; // Translate index to tree index
        var change = priority - _tree[treeIndex];
        _tree[treeIndex] = priority;

        if (priority > MaxPriority)
        {
            MaxPriority = priority;
        }

        // Update parent nodes
        while (treeIndex > 0)
        {
            treeIndex /= 2; // Move up the tree
            _tree[treeIndex] += change; // Update the sum
        }
    }
    
    public int GetMinPriorityIndex()
    {
        var minPriorityIndex = 0;
        var minPriority = float.MaxValue;

        for (var i = capacity; i < 2 * capacity; i++)
        {
            if (_tree[i] < minPriority)
            {
                minPriority = _tree[i];
                minPriorityIndex = i - capacity;
            }
        }

        return minPriorityIndex;
    }
    
    public int GetMaxPriorityIndex()
    {
        var maxPriorityIndex = 0;
        var maxPriority = float.MinValue;

        for (var i = capacity; i < 2 * capacity; i++)
        {
            if (_tree[i] > maxPriority)
            {
                maxPriority = _tree[i];
                maxPriorityIndex = i - capacity;
            }
        }

        return maxPriorityIndex;
    }

    public int Retrieve(float value)
    {
        var index = 1; // Start from the root
        while (index < capacity) // Ensure index is within leaf node range
        {
            index *= 2; // Move to left child by default
            if (value >= _tree[index])
            {
                value -= _tree[index]; // Deduct the value of the left child when moving right
                index++; // Move to right child
            }
        }
        
        return Math.Min(index - capacity, _count - 1); // Ensure index is within bounds
    }


    public int Add(float priority)
    {
        // Instead of throwing an exception, recycle the oldest entry
        var index = _count % capacity; // Use modulo to create a circular buffer effect
        Update(index, priority); // Update or add the priority in the SumTree
        if (_count < capacity) _count++; // Increment count until the tree is full
        return index;
    }

    public void Clear()
    {
#if NET8_0_OR_GREATER
        Array.Fill(_tree, 0f); // Reset all tree elements to 0
#else
        for (var i = 0; i < _tree.Length; i++)
        {
            _tree[i] = 0f;
        }
#endif
        _count = 0; // Reset count
        MaxPriority = 1.0f; // Reset MaxPriority
    }
}