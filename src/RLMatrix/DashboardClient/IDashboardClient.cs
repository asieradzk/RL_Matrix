namespace RLMatrix;

/// <summary>
///		Defines the contract for dashboard clients.
/// </summary>
public interface IDashboardClient : IAsyncDisposable
{
	/// <summary>
	///		Adds a new data point to the experiment data collection.
	/// </summary>
	/// <param name="data">The new experiment data point.</param>
	Task AddDataPointAsync(ExperimentData data);

	/// <summary>
	///		Saves the model to the specified path.
	/// </summary>
	/// <param name="path">The path to save to.</param>
	/// <returns></returns>
	Task SaveModelAsync(string path);

	/// <summary>
	///		Loads the model from the specified path.
	/// </summary>
	/// <param name="path">The path to load from.</param>
	/// <returns></returns>
	Task LoadModelAsync(string path);

	/// <summary>
	///		Saves the buffer to the specified path.
	/// </summary>
	/// <param name="path">The path to save to.</param>
	/// <returns></returns>
	Task SaveBufferAsync(string path);
	
	public void UpdateEpisodeData(float? reward, float? cumReward, int? epLength);
	public void UpdateActorLoss(float? loss);
	public void UpdateActorLearningRate(float? lr);
	public void UpdateCriticLoss(float? loss);
	public void UpdateCriticLearningRate(float? lr);
	public void UpdateKLDivergence(float? kl);
	public void UpdateEntropy(float? entropy);
	public void UpdateEpsilon(float? epsilon);
	public void UpdateLoss(float? loss);
	public void UpdateLearningRate(float? lr);
}