import numpy as np
import threading
from mlagents.torch_utils import torch
from mlagents.trainers.torch.model_serialization import TensorNames
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.settings import SerializationSettings


logger = get_logger(__name__)



class exporting_to_onnx:
    """
    Set this context by calling
    ```
    with exporting_to_onnx():
    ```
    Within this context, the variable exporting_to_onnx.is_exporting() will be true.
    This implementation is thread safe.
    """

    # local is_exporting flag for each thread
    _local_data = threading.local()
    _local_data._is_exporting = False

    # global lock shared among all threads, to make sure only one thread is exporting at a time
    _lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        self._local_data._is_exporting = True

    def __exit__(self, *args):
        self._local_data._is_exporting = False
        self._lock.release()

    @staticmethod
    def is_exporting():
        if not hasattr(exporting_to_onnx._local_data, "_is_exporting"):
            return False
        return exporting_to_onnx._local_data._is_exporting
    
class SACOnnxablePolicy(torch.nn.Module):
    def __init__(self,  actor):
        super(SACOnnxablePolicy, self).__init__()
        
        # Removing the flatten layer because it can't be onnxed
        self.actor = torch.nn.Sequential(actor.latent_pi, actor.mu)

    def forward(self, input_):

        #skill_0 = torch.zeros((input_.shape[0],1),dtype=input_.dtype)
        #skill_1 = torch.ones((input_.shape[0],1),dtype=input_.dtype)
        
        #input_ = torch.cat([input_,skill_0,skill_1],axis=1)
        export_out = []
        version = torch.nn.Parameter(torch.Tensor([2.]), requires_grad=False)
        memory = torch.nn.Parameter(torch.Tensor([0]), requires_grad=False)
        is_continuous_control = torch.nn.Parameter(torch.Tensor([1]), requires_grad=False)
        export_out += [version, memory]
        action = self.actor(input_).detach()
        action_sum = action.shape[-1].item()
        action_size =  torch.nn.Parameter(torch.Tensor([action_sum]),requires_grad=False)
        export_out += [action, action_size]
        export_out += [action]
        export_out += [is_continuous_control]     
        export_out += [action_size]
        #print(export_out)
        return tuple(export_out)
        #return action
    
class ModelSerializer:
    def __init__(self, policy):
        # ONNX only support input in NCHW (channel first) format.
        # Barracuda also expect to get data in NCHW.
        # Any multi-dimentional input should follow that otherwise will
        # cause problem to barracuda import.
        self.policy = policy
        self.actor = SACOnnxablePolicy(self.policy.actor)
        #observation_specs = self.policy.behavior_spec.observation_specs
        obs_size = self.policy.observation_space.shape[0]+self.policy.prior.event_shape[0]
        observation_specs = [np.zeros((obs_size,))]
        batch_dim = [1]
        seq_len_dim = [1]
        vec_obs_size = 0
        
        for obs_spec in observation_specs:
            if len(obs_spec.shape) == 1:
                vec_obs_size += obs_spec.shape[0]
        num_vis_obs = sum(
            1 for obs_spec in observation_specs if len(obs_spec.shape) == 3
        )
        dummy_vec_obs = [torch.zeros(batch_dim + [vec_obs_size])]
        # create input shape of NCHW
        # (It's NHWC in observation_specs.shape)
        dummy_vis_obs = [
            torch.zeros(
                batch_dim + [obs_spec.shape[2], obs_spec.shape[0], obs_spec.shape[1]]
            )
            for obs_spec in observation_specs
            if len(obs_spec.shape) == 3
        ]

        dummy_var_len_obs = [
            torch.zeros(batch_dim + [obs_spec.shape[0], obs_spec.shape[1]])
            for obs_spec in observation_specs
            if len(obs_spec.shape) == 2
        ]

        self.dummy_input = (
            dummy_vec_obs,
            dummy_vis_obs,
            dummy_var_len_obs,
        )

        self.input_names = [TensorNames.vector_observation_placeholder]
        for i in range(num_vis_obs):
            self.input_names.append(TensorNames.get_visual_observation_name(i))
        for i, obs_spec in enumerate(observation_specs):
            if len(obs_spec.shape) == 2:
                self.input_names.append(TensorNames.get_observation_name(i))


        self.dynamic_axes = {name: {0: "batch"} for name in self.input_names}

        self.output_names = [TensorNames.version_number, TensorNames.memory_size]
    
        action_spec_cont = policy.action_space
        action_spec_disc = np.zeros(0)
        if action_spec_cont.shape[0] > 0:
            self.output_names += [
                TensorNames.continuous_action_output,
                TensorNames.continuous_action_output_shape,
            ]
            self.dynamic_axes.update(
                {TensorNames.continuous_action_output: {0: "batch"}}
            )
        if action_spec_disc.shape[0] > 0:
            self.output_names += [
                TensorNames.discrete_action_output,
                TensorNames.discrete_action_output_shape,
            ]
            self.dynamic_axes.update({TensorNames.discrete_action_output: {0: "batch"}})
        if (
            action_spec_cont.shape[0] == 0
            or action_spec_disc.shape[0] == 0
        ):
            self.output_names += [
                TensorNames.action_output_deprecated,
                TensorNames.is_continuous_control_deprecated,
                TensorNames.action_output_shape_deprecated,
            ]
            self.dynamic_axes.update(
                {TensorNames.action_output_deprecated: {0: "batch"}}
            )
            

    def export_policy_model(self, output_filepath: str) -> None:
        """
        Exports a Torch model for a Policy to .onnx format for Unity embedding.

        :param output_filepath: file path to output the model (without file suffix)
        """
        onnx_output_path = f"{output_filepath}.onnx"
        logger.info(f"Converting to {onnx_output_path}")
        print(self.output_names)
        print(len(self.dummy_input[0][0]))
        print(self.dummy_input[0][0].shape)
        with exporting_to_onnx():
            torch.onnx.export(
                self.actor,
                self.dummy_input[0][0],
                onnx_output_path,
                opset_version=9,
                input_names=self.input_names,
                output_names=self.output_names,
                #output_names=['continuous_actions'],
                dynamic_axes=self.dynamic_axes,
                verbose=1

            )
        logger.info(f"Exported {onnx_output_path}")