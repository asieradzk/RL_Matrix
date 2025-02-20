using MessagePack;
using RLMatrix.Common;

namespace RLMatrix.Remote;

[MessagePackObject]
public class StateSizesDTO(int? singleValue, int? value1, int? value2)
{
    [Key(0)]
    public int? SingleValue { get; set; } = singleValue;

    [Key(1)]
    public int? Value1 { get; set; } = value1;

    [Key(2)]
    public int? Value2 { get; set; } = value2;

    internal static StateSizesDTO Create1D(int value)
    {
        return new(value, null, null);
    }

    internal static StateSizesDTO Create2D(int value1, int value2)
    {
        return new(null, value1, value2);
    }
}

public static class StateSizesDTOExtensions
{
    public static StateSizesDTO ToStateSizesDTO(this StateDimensions value)
    {
        return value.Dimensions.Length switch
        {
            1 => StateSizesDTO.Create1D(value.Dimensions[0]),
            2 => StateSizesDTO.Create2D(value.Dimensions[1], value.Dimensions[2]),
            _ => throw new ArgumentOutOfRangeException(nameof(value.Dimensions))
        };
    }

    public static StateDimensions ToStateDimensions(this StateSizesDTO dto)
    {
        if (dto.SingleValue.HasValue)
        {
            return dto.SingleValue.Value;
        }
        
        if (dto is { Value1: { } value1, Value2: { } value2 })
        {
            return (value1, value2);
        }
        
        throw new ArgumentException("Invalid StateSizesDTO: both SingleValue and tuple values are null.");
    }
}

