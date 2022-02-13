using Onnx.ShuffleNet;

if(args.Length == 0)
{
    throw new ArgumentNullException("No input was given");
}

const string ModelFilePath = "shufflenet-v2-12.onnx";
const string ModelInputName = "input";

var preProcess = new PreProcessing(ModelInputName);
var inputs = preProcess.PrepareInputs(args[0]);

var inference = new Inference(ModelFilePath);
var result = inference.Run(inputs);

var top3 = result.Select((x, i) => new InferenceResult { Label = LabelMap.Labels[i], Score = x })
               .OrderByDescending(x => x.Score)
               .Take(3);

Console.WriteLine("Top 3 predictions");

foreach (var t in top3)
{
    Console.WriteLine($"Label: {t.Label}, Score: {t.Score}");
}