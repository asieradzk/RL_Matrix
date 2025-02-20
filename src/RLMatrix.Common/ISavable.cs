namespace RLMatrix.Common;

/// <summary>
///     Defines an interface for objects that can be saved and loaded.
/// </summary>
public interface ISavable
{
    /// <summary>
    ///     Saves the object to the specified path.
    /// </summary>
    /// <param name="path">The path to save the object to.</param>
    ValueTask SaveAsync(string path);

    /// <summary>
    ///     Loads the object from the specified path.
    /// </summary>
    /// <param name="path">The path to load the object from.</param>
    ValueTask LoadAsync(string path);
}