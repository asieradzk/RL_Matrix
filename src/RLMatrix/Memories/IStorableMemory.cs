namespace RLMatrix;

public interface IStorableMemory
{
    void Save(string path);
    void Load(string path);
}