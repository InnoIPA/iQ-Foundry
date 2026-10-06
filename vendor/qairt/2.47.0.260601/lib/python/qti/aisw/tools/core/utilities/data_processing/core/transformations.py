import glob
import importlib
import inspect
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel
from qti.aisw.tools.core.utilities.data_processing.core.adapters import OutputAdapter
from qti.aisw.tools.core.utilities.data_processing.core.representations import Representation
from qti.aisw.tools.core.utilities.data_processing.datasets import IndexableDataset
from qti.aisw.tools.core.utilities.data_processing.metrics import Metric


class Transformation(ABC):
    """Base class for all Transformations.

    This class provides a common interface for all transformations.
    It includes an execute method that takes in a data representation and
    returns the processed data with updated metadata.
    The execute_index method is used to process multiple data samples together.
    """

    def __init__(self, **kwargs):
        """Initializes the Transformation Base class with optional arguments.

        Args:
            kwargs: Optional keyword-based arguments that modify the execution logic
        """
        # Implement specific transformation initialization logic here
        self.__dict__.update(kwargs)
        self.validate()

    def validate(self):
        """Validates the Transformation class."""
        # Implement specific validation logic here
        pass

    def validate_input(self, input_sample: Representation) -> Representation:
        """Validates the input sample before processing it with the Transformation.
        This method is used to validate the input data before calling the execute method.

        Args:
           input_sample: Data representation that needs validation.

        Returns:
           Representation: Data representation post the input validation
        """
        return input_sample

    def validate_output(self, output_sample: Representation) -> Representation:
        """Validates the output sample before returning it from the Transformation.
        This method is used to validate the output data before returning it from the Transformation.

        Args:
           output_sample: Data representation that needs validation.

        Returns:
           Representation: Data representation post the output validation
        """
        return output_sample

    @staticmethod
    def validate_input_output(execute) -> "execute":
        """This decorator function is used to validate the input data before
        calling the execute method. Output data is also validated after calling
        execute method.

        Args:
            execute: Method being decorated.

        Returns:
            Decorated execute method.
        """

        def wrap(self, input_sample):
            validated_input = self.validate_input(input_sample)
            output = execute(self, validated_input)
            validated_output = self.validate_output(output)
            return validated_output

        return wrap

    @validate_input_output
    @abstractmethod
    def execute(self, data: Representation) -> Representation:
        """Execute method for the Transformation class.

        Args:
            data (Representation): Data representation containing the data and metadata

        Returns:
            Representation: Processed data representation with updated metadata
        """
        # Operate on a single data sample
        raise NotImplementedError("Each transformation must implement the execute method.")

    def execute_multiple_samples(self, data: list[Representation]) -> list[Representation]:
        """Execute method for processing multiple data samples together.

        Args:
            data (list[Representation]): list of data representations containing individual
             data samples

        Returns:
            list[Representation]: Processed list of data representations with updated metadata

        Note: This is to be used in scenarios where the processor logic depends on more than
        one model inputs/outputs for e.g. In HRNET Postprocesssing, we require the model
        outputs for both flipped and actual image to compute the postprocessing step.
        """
        # Default implementation is to loop over the representation and operate on each data sample
        result = []
        for d in data:
            result.append(self.execute(d))
        return result

    def __call__(
        self, data: Representation | list[Representation], **kwargs
    ) -> Representation | list[Representation]:
        """Invokes the transformation on a single or multiple data samples.

        Args:
            data (Representation | list[Representation]) : Single data representation
                                    or list of data representations containing individual data samples
            kwargs: Optional keyword-based arguments that modify the execution logic

        Returns:
            Representation | list[Representation: Processed data representation with updated metadata
        """
        try:
            if isinstance(data, list):
                result = self.execute_multiple_samples(data, **kwargs)
            else:
                result = self.execute(data, **kwargs)
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to execute transformation: {self.__class__.__name__}. Reason : {e}")


class PreProcessor(Transformation):
    """Pre-processing transformation base class.

    Attributes:
        _transformation_type (str): Type of the transformation
    """

    _transformation_type = "preproc"


class PostProcessor(Transformation):
    """Post-processing transformation base class.

    Attributes:
        _transformation_type (str): Type of the transformation
    """

    _transformation_type = "postproc"


class ComponentConfig(AISWBaseModel):
    """A class representing a configuration object.

    Args:
        name (str): The name of the configuration.
        params (Optional[Dict[str, Any]], optional): The parameters of the configuration. Defaults to None.
    """

    name: str
    params: Optional[dict] = None
    _type: Literal["preprocessor", "postprocessor", "metric", "dataset", "adapter"] = None

    def model_post_init(self, __context):
        """Validate component configuration after initialization.

        This method is called by Pydantic's `model_post_init` hook.
        It ensures that the component configuration conforms to the expected structure and type.

        Args:
            self (ComponentConfig): The current instance of ComponentConfig.
            __context: The context in which this method was called (not used in this implementation).
        """
        if self._type:
            ComponentRegistry().validate_component(
                component_name=self.name, component_params=self.params, component_type=self._type
            )


class PreProcessorConfig(ComponentConfig):
    """Defines parameters expected in a PreProcessor.
    This class defines the configuration parameters required for a PreProcessor.

    Attributes:
        name (str): The name of the PreProcessor.
        params (dict): A dictionary of parameter-value pairs for the PreProcessor.
    """

    _type = "preprocessor"


class PostProcessorConfig(ComponentConfig):
    """Defines parameters expected in a PostProcessor.
    This class defines the configuration parameters required for a PostProcessor.

    Attributes:
        name (str): The name of the PostProcessor.
        params (dict): A dictionary of parameter-value pairs for the PostProcessor.
    """

    _type = "postprocessor"


class AdapterConfig(ComponentConfig):
    """Defines parameters expected in an Adapter.
    This class defines the configuration parameters required for an Adapter.

    Attributes:
        name (str): The name of the Adapter.
        params (dict): A dictionary of parameter-value pairs for the Adapter.
    """

    _type = "adapter"


class MetricConfig(ComponentConfig):
    """Defines parameters expected in a Metric.
    This class defines the configuration parameters required for a Metric.

    Attributes:
        name (str): The name of the Metric.
        params (dict): A dictionary of parameter-value pairs for the Metric.
    """

    _type = "metric"


class DatasetConfig(ComponentConfig):
    """Defines parameters expected in a Dataset.
    This class defines the configuration parameters required for a Dataset.

    Attributes:
        name (str): The name of the dataset.
        params (dict): A dictionary of parameter-value pairs for the dataset.

    """

    _type = "dataset"


class ComponentRegistry:
    """Central registry for all components."""

    registered_preprocessors = {}
    registered_postprocessors = {}
    registered_metrics = {}
    registered_datasets = {}
    registered_adapters = {}
    _instance = None

    def __new__(cls) -> "ComponentRegistry":
        """Get or create a singleton instance of the component registry."""
        if cls._instance is None:
            cls._instance = super(ComponentRegistry, cls).__new__(cls)
            cls._instance.find_components()
        return cls._instance

    @classmethod
    def find_components(cls) -> None:
        """Find all available components and register them."""
        if not hasattr(cls, "_components_found"):
            cls._components_found = True
        components_base_dir = str(Path(__file__).resolve().parents[1])
        abs_path = os.path.abspath(components_base_dir)
        sys.path.append(abs_path)

        # Add directories to sys path
        dirs = []
        for dir in os.walk(abs_path):
            if dir[0] not in dirs:
                sys.path.append(dir[0])
                dirs.append(dir[0])

        # search components recursively
        paths = Path(components_base_dir).rglob('*.py')
        files = [path.name for path in paths if path.is_file()]
        for idx, file in enumerate(files):
            if file.endswith(".py"):
                if file.startswith("__") or file.startswith("test"):
                    continue
                file = os.path.splitext(file)[0]
                _plugin = importlib.import_module(file)
                classes = inspect.getmembers(_plugin, predicate=inspect.isclass)
                for cl in classes:
                    if cl[1].__module__ == file:
                        class_hier = inspect.getmro(cl[1])
                        for class_h in class_hier:
                            if class_h.__name__ == "PreProcessor":
                                cls.registered_preprocessors[cl[0]] = cl[1]
                                break
                            elif class_h.__name__ == "PostProcessor":
                                cls.registered_postprocessors[cl[0]] = cl[1]
                                break
                            elif class_h.__name__ == "Metric":
                                cls.registered_metrics[cl[0]] = cl[1]
                                break
                            elif class_h.__name__ == "IndexableDataset":
                                cls.registered_datasets[cl[0]] = cl[1]
                                break
                            elif class_h.__name__ == "OutputAdapter":
                                cls.registered_adapters[cl[0]] = cl[1]
                                break

    @classmethod
    def register_component(
        cls,
        component_name: str,
        component_cls: object,
        component_type: Literal["preprocessor", "postprocessor", "metric", "dataset", "adapter"],
    ) -> None:
        """Register a component based on its type.

        Args:
            cls (Type['Transformations']): The class that is registering the component.
            component_name (str): The name of the registered component.
            component_cls (object): The class of the component being registered.
            component_type (Literal['processor', 'metric', 'dataset', 'adapter']):
                The type of the component.

        Raises:
            ValueError: If the component type is invalid.

        Notes:
            This method registers a component with the provided name and class, based on its type.
            Supported types are processor, metric, dataset, and adapter.
        """
        if component_type not in ["preprocessor", "postprocessor", "metric", "dataset", "adapter"]:
            raise ValueError(
                f"The component type '{component_type}' is not supported."
                "Supported are preprocessor, postprocessor, metric, dataset and adapter."
            )
        if component_type == "processor" and isinstance(component_cls, PreProcessor):
            cls.registered_preprocessors[component_name] = component_cls
        elif component_type == "adapter" and isinstance(component_cls, OutputAdapter):
            cls.registered_adapters[component_name] = component_cls
        elif component_type == "postprocessor" and isinstance(component_cls, PostProcessor):
            cls.registered_postprocessors[component_name] = component_cls
        elif component_type == "dataset" and isinstance(component_cls, IndexableDataset):
            cls.registered_dataset_plugins[component_name] = component_cls
        elif component_type == "metric" and isinstance(component_cls, Metric):
            cls.registered_metrics[component_name] = component_cls
        else:
            raise ValueError(f"Invalid component type {component_type}")

    @classmethod
    def validate_component(
        cls,
        component_name: str,
        component_params: dict,
        component_type: Literal["preprocessor", "postprocessor", "metric", "dataset", "adapter"],
    ) -> bool:
        """Validate that a component of a given type exists in the registered components dictionary.

        Args:
            component_name (str): The name of the component to validate.
            component_params (dict): The parameters to use when instantiating the component.
            component_type (Literal['processor', 'metric', 'dataset', 'adapter']):
                The type of the component.

        Returns:
            bool: True if the component is valid, False otherwise.

        Raises:
            ValueError: If the component type is not registered.
            RuntimeError: If instantiation of the component fails.
        """
        # Set registered_components to the corresponding class dictionary attribute based on component_type
        try:
            registered_components = getattr(cls, f"registered_{component_type}s")
        except AttributeError:
            raise ValueError(
                f"The component type '{component_type}' is not supported."
                "Supported are preprocessor, postprocessor, metric, dataset and adapter."
            )

        if component_name not in registered_components:
            raise ValueError(f"{component_name} of type {component_type} is not available/registered")

        # Get the component class from the dictionary
        component_class = registered_components[component_name]
        try:
            # Try to instantiate the component with the given parameters
            if component_params:
                component_class(**component_params)
            else:
                component_class()
        except Exception as e:
            # If instantiation fails, raise an error
            raise RuntimeError(f"Failed to create {component_name} of type {component_type}: {str(e)}")
        return True

    @classmethod
    def get_components_from_configs(
        cls, component_configs: list[ComponentConfig], use_calibration: bool = False, max_samples=None
    ):
        """Returns a list of components from the given configurations."""
        component_objs = []
        for component_config in component_configs:
            if cls.validate_component(
                component_name=component_config.name,
                component_params=component_config.params,
                component_type=component_config._type,
            ):
                registered_components = getattr(cls, f"registered_{component_config._type}s")
                component_class = registered_components[component_config.name]
                if component_config._type == "dataset":
                    if max_samples is not None:
                        component_config.params["max_samples"] = max_samples
                    component_config.params["use_calibration"] = use_calibration
                component_objs.append(
                    component_class(**component_config.params)
                    if component_config.params
                    else component_class()
                )
        return component_objs


class ProcessorChainExecutor:
    """Executes a chain of processors on input samples.

    Args:
        processors: List of preprocessors or postprocessors to execute in sequence.
        dump_outputs: Whether to save the output of each processor (default: False).
        output_dir: Directory where output files are saved (default: None).

    Attributes:
        processors (list): List of preprocessors or postprocessors.
    """

    def __init__(
        self,
        processors: list[PreProcessor | PostProcessor],
        dump_outputs=False,
        output_dir: os.PathLike = None,
        node_names: list[str] = None,
        dump_filelist=False,
    ):
        """Initializes the transformation processor with a list of preprocessors and postprocessors.

        Args:
            processors (list[PreProcessor | PostProcessor]): A list of preprocessors and
             postprocessors to be applied in sequence.
            dump_outputs (bool, optional): Whether to dump the outputs. Defaults to False.
            output_dir (os.PathLike, optional): The directory where the outputs will be dumped.
                If `dump_outputs` is True, this must not be None. Defaults to None.
            node_names (list[str], optional): The names of the nodes in the model graph.
                 Defaults to None.
            dump_filelist (bool, optional): Whether to dump the filelist of the processed inputs.
                 Defaults to False.
        """
        self._processors = processors
        self._output_dir = output_dir
        self._dump_outputs = dump_outputs
        self.node_names = node_names
        self._dump_filelist = dump_filelist
        if self._dump_outputs:
            Path(self._output_dir).mkdir(exist_ok=True)
            if self._dump_filelist:
                self.file_list_handle = open(os.path.join(self._output_dir, 'processed-outputs.txt'), 'w')
        self.validate()

    def validate(self):
        """Checks if the required parameters are set.

        Raises:
            ValueError: If the output directory is not specified and outputs are to be dumped.
        """
        if not isinstance(self._processors, list):
            raise ValueError("Processors provided must be a list")
        if self._dump_outputs and not self._output_dir:
            raise ValueError("Output directory is required to dump outputs")
        if self.node_names and not any(isinstance(node_name, str) for node_name in self.node_names):
            raise ValueError("Node names must be strings")

        if self._dump_filelist and not self._dump_outputs:
            raise ValueError("Dumping filelist requires dumping of outputs")

    def process(self, input_sample: Representation):
        """Executes all the transformations in the processors chain to process input samples.

        Args:
            input_sample : input sample to be processed.
        """
        for processor in self._processors:
            input_sample = processor(input_sample)
        output_paths_per_sample = None
        output_paths_per_sample_str = ''
        if self._dump_outputs:
            output_paths_per_sample = self.save_outputs(input_sample)
            output_paths_per_sample_str = ",".join(output_paths_per_sample) + "\n"
        return input_sample, output_paths_per_sample_str

    def save_outputs(self, sample: Representation, sub_folder_name: str = ""):
        """Saves the outputs of a transformation for each data point in the sample.

        Args:
            sample (Representation): The input sample to save outputs for.
            sub_folder_name (str, optional): Sub-folder name within the output directory.
             Defaults to an empty string.

        Returns:
            list[str]: A list of file paths where the outputs were saved.
        """
        output_paths_per_sample = []
        for data_idx in range(len(sample.data)):
            sample_idx = sample.idx if sample.idx is not None else ""
            if self.node_names:
                out_filename = f"{self.node_names[data_idx]}_{sample_idx}.raw"
                abs_path = os.path.join(self._output_dir, out_filename)
                if self._dump_filelist:
                    if data_idx:
                        self.file_list_handle.write(f" {self.node_names[data_idx]}:={abs_path}")
                    else:
                        self.file_list_handle.write(f"{self.node_names[data_idx]}:={abs_path}")
            else:
                out_filename = f"{sample._transformation_type}essed_{sample_idx}_{data_idx}.raw"
                abs_path = os.path.join(self._output_dir, out_filename)
                if self._dump_filelist:
                    self.file_list_handle.write(f"{abs_path},")
            if self._dump_filelist:
                self.file_list_handle.write("\n")
                self.file_list_handle.flush()
            output_paths_per_sample.append(abs_path)
        sample.save(output_paths_per_sample)
        return output_paths_per_sample
