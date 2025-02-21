namespace RLMatrix;

public interface IStateActionValuesExtractor
{
    Tensor ExtractStateActionValues(Tensor qValues, Tensor actions);
}