import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner, Evaluator, Pusher
from tfx.proto import example_gen_pb2, trainer_pb2, transform_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration import pipeline, metadata
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

from modules.tuner import masked_accuracy

PIPELINE_NAME = 'machine-translation-id-en-pipeline'
SCHEMA_PIPELINE_NAME = 'machine-translation-id-en-schema'
PIPELINE_ROOT = os.path.join('machine-learning', 'pipelines', PIPELINE_NAME)
METEDATA_PATH = os.path.join('machine-learning', 'metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL = os.path.join('machine-learning', 'serving_model', PIPELINE_NAME)
DATA_ROOT = os.path.join('machine-learning', 'data', 'final')

output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
        example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
    ])
)
example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output)
statistic_gen = StatisticsGen(examples=example_gen.outputs['examples'])
schema_gen = SchemaGen(statistics=statistic_gen.outputs['statistics'])
example_validator = ExampleValidator(
    statistics=statistic_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.join('machine-learning', 'modules', 'tranform.py')
)
tuner = Tuner(
    module_file=os.path.abspath(os.path.join('machine-learning','modules', 'tuner.py')),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],   
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval']),
)
trainer = Trainer(
    module_file=os.path.abspath(os.path.join('machine-learning','modules', 'trainer.py')),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval']),
)

model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')

class MaskedAccuracy(tfma.metrics.Metric):
    def __init__(self):
        super().__init__(
            metric_fn=masked_accuracy,
            name='masked_accuracy',
            model_names=[''],
            output_names=[''],
            example_weighted=False
        )

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='tags_xf')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(
                class_name='MaskedAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.5}
                    ),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': 0.01}
                    )
                )
            )
        ])
    ]
)

evaluator = Evaluator(
    examples=transform.outputs['transformed_examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config
)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=SERVING_MODEL
        )
    )
)

def create_pipeline():
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=[
            example_gen,
            statistic_gen,
            schema_gen,
            example_validator,
            transform,
            tuner,
            trainer,
            model_resolver,
            evaluator,
            pusher
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METEDATA_PATH)
    )

if __name__ == '__main__':
    LocalDagRunner().run(create_pipeline())