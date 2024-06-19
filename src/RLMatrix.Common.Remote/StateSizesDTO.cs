using MessagePack;
using OneOf;

namespace RLMatrix.Common.Remote;

[MessagePackObject]
public class StateSizesDTO
{
    [Key(0)]
    public int? SingleValue { get; set; }

    [Key(1)]
    public int? Value1 { get; set; }

    [Key(2)]
    public int? Value2 { get; set; }
}

public static class StateSizesDTOExtensions
{
    public static StateSizesDTO ToStateSizesDTO(this OneOf<int, (int, int)> value)
    {
        return value.Match(
            singleValue => new StateSizesDTO
            {
                SingleValue = singleValue,
                Value1 = null,
                Value2 = null
            },
            tupleValue => new StateSizesDTO
            {
                SingleValue = null,
                Value1 = tupleValue.Item1,
                Value2 = tupleValue.Item2
            }
        );
    }

    public static OneOf<int, (int, int)> ToOneOf(this StateSizesDTO dto)
    {
        if (dto.SingleValue.HasValue)
        {
            return dto.SingleValue.Value;
        }
        else if (dto.Value1.HasValue && dto.Value2.HasValue)
        {
            return (dto.Value1.Value, dto.Value2.Value);
        }
        else
        {
            throw new ArgumentException("Invalid StateSizesDTO: both SingleValue and tuple values are null.");
        }
    }
}

