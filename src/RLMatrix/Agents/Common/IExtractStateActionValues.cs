namespace RLMatrix;

public interface IExtractStateActionValues
{
    Tensor ExtractStateActionValues(Tensor qValues, Tensor actions);
}