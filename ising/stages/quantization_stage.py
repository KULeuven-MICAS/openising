from ising.stages import LOGGER
from typing import Any
import numpy as np
import math
import copy
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel

class QuantizationStage(Stage):
    """! Stage to quantize the Ising model."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 config: Any,
                 ising_model: IsingModel | None = None,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.ising_model_ori = ising_model

        ##############################################
        ## For testing purpose: built-in parameters
        ##############################################
        self.shift_ising_model = False  # True: shift the ising model to center at zero
        self.visualize_J_matrix = False  # True: visualize the J matrix

        if self.shift_ising_model:
            # shift J in ising model centraling at zero
            weight = copy.deepcopy(self.ising_model_ori.J)
            nonzero_mask = weight != 0
            weight_min = np.min(weight)
            shift_bias = weight_min / 2
            shifted_weight = copy.deepcopy(weight)
            shifted_weight[nonzero_mask] = shifted_weight[nonzero_mask] - shift_bias
            self.ising_model = IsingModel(
                J=shifted_weight,
                h=self.ising_model_ori.h,
                c=self.ising_model_ori.c,
            )
        else:
            self.ising_model = copy.deepcopy(ising_model)

        # visualize the J matrix
        if self.visualize_J_matrix:
            if self.shift_ising_model:
                self.plot_ndarray_in_matrix(shifted_weight)
            else:
                self.plot_ndarray_in_matrix(self.ising_model.J)
        else:
            LOGGER.debug("J matrix visualization is disabled.")

    def run(self) -> Any:
        """! Quantize the J of the Ising model."""
        if self.config.quantization:
            quantization_precision = self.config.quantization_precision
            original_J = self.ising_model.J
            quantized_J = self.quantize_J_matrix(original_J, quantization_precision)

            original_h = self.ising_model.h
            quantized_model = IsingModel(
                J=quantized_J,
                h=original_h,
                c=self.ising_model.c,
            )
        else:
            LOGGER.debug("Quantization is disabled.")
            quantized_model = copy.deepcopy(self.ising_model)

        self.kwargs["config"] = self.config
        self.kwargs["ising_model"] = quantized_model
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for ans, debug_info in sub_stage.run():
            ans.ising_model = self.ising_model
            ans.quantized_model = quantized_model
            for energy_id in range(len(ans.energies)):
                ans.energies[energy_id] = self.ising_model.evaluate(ans.states[energy_id])
                if hasattr(ans, "tsp_energies"):
                    if ans.tsp_energies[energy_id] == math.inf:
                        ans.tsp_energies[energy_id] = math.inf
                    else:
                        ans.tsp_energies[energy_id] = ans.energies[energy_id]
            yield ans, debug_info

    def quantize_J_matrix(self, J: np.ndarray, quantization_precision: int | float = 2) -> np.ndarray:
        """! Quantizes the J matrix to a given precision.

        @param J: the input J matrix
        @param quantization_precision: the precision for quantization

        @return: a quantized J matrix
        """
        J_min = int(np.min(J))
        J_max = int(np.max(J))
        original_required_int_precision = math.ceil(math.log2(J_max - J_min))
        LOGGER.info(
            "Original required int precision: %s, current quant: %s",
            original_required_int_precision,
            quantization_precision,
        )
        assert quantization_precision <= original_required_int_precision, \
            f"Quantization precision {quantization_precision} is larger " \
            f"than the original precision {original_required_int_precision}."
        assert quantization_precision == 1.5 or isinstance(quantization_precision, int)
        if quantization_precision == 1.5:
            step_size = (J_max - J_min) / 2
            nonzero_mask = J != 0
            quantized_J = copy.deepcopy(J)
            quantized_J[nonzero_mask] = np.round((J[nonzero_mask] - J_min) / step_size) * step_size + J_min
        else:
            step_size = (J_max - J_min) / (2 ** quantization_precision - 1)
            nonzero_mask = J != 0
            quantized_J = copy.deepcopy(J)
            quantized_J[nonzero_mask] = np.round((J[nonzero_mask] - J_min) / step_size) * step_size + J_min
        self.plot_ndarray_in_matrix(quantized_J)
        return quantized_J

    @staticmethod
    def plot_ndarray_in_matrix(mat: np.ndarray, output: str = "vdarray_matrix.png"):
        """ ! Visualize 2D ndarray in matrix
        @param mat: input 2D ndarray
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(1, 1)
        sns.heatmap(mat,
                    ax=ax,
                    cmap="viridis",  # Yellow-Orange-Red colormap
                    cbar_kws={"label": "Value"})
        # Add colorbar legend
        ax.figure.axes[-1].yaxis.label.set_size(12)
        ax.set_title(
        f"Shape: {mat.shape}, value min: {np.min(mat)}, max: {np.max(mat)},"
        f"mean: {round(np.mean(mat), 2)}, unique levels: {len(np.unique(mat))}",
            loc="left",
            pad=10, weight="bold", fontsize=8)
        ax.set_xlabel("ID", fontsize=12, weight="bold")
        ax.set_ylabel("ID", fontsize=12, weight="bold")
        # Add box around the subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)  # Adjust the line width of the box
            spine.set_color("black")  # Set the color of the box
        plt.tight_layout()
        plt.savefig(output)
