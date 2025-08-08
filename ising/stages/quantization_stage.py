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
        original_required_int_precision = self.calc_original_precision(self.ising_model.J)
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
            ans.original_required_int_precision = original_required_int_precision
            for energy_id in range(len(ans.energies)):
                ans.energies[energy_id] = self.ising_model.evaluate(ans.states[energy_id])
                if hasattr(ans, "tsp_energies"):
                    if ans.tsp_energies[energy_id] == math.inf:
                        ans.tsp_energies[energy_id] = math.inf
                    else:
                        ans.tsp_energies[energy_id] = ans.energies[energy_id]
            yield ans, debug_info

    def calc_original_precision(self, J: np.ndarray) -> int:
        """! Calculate the original precision of the J matrix.

        @param J: the input J matrix

        @return: the original required int precision
        """
        J_min = int(np.min(J))
        J_max = int(np.max(J))
        if (J_min >= 0 and J_max >= 0) or (J_min <= 0 and J_max <= 0):
            same_sign = True
        else:
            same_sign = False
        if same_sign:
            # If J has only positive or only negative values, we can calculate the precision
            # based on the range from 0 to the maximum value.
            maximum_value = max(abs(J_min), abs(J_max))
            # The range is (0 to J_max), and we add 1 to ensure we cover the full range.
            original_required_int_precision = math.ceil(math.log2(maximum_value + 1))
        else:
            # If J has both positive and negative values, we need to consider the range
            # from the minimum to the maximum value.
            # The range is (J_max - J_min), and we add 1 to ensure we cover the full range.
            # This is because we need to represent both positive and negative values.
            original_required_int_precision = math.ceil(math.log2(abs(J_max - J_min) + 1))
        return original_required_int_precision

    def quantize_J_matrix(self, J: np.ndarray, quantization_precision: int | float = 2) -> np.ndarray:
        """! Quantizes the J matrix to a given precision.

        @param J: the input J matrix
        @param quantization_precision: the precision for quantization

        @return: a quantized J matrix
        """
        J_min = int(np.min(J))
        J_max = int(np.max(J))
        if (J_min >= 0 and J_max >= 0) or (J_min <= 0 and J_max <= 0):
            same_sign = True
        else:
            same_sign = False
        original_required_int_precision = self.calc_original_precision(J)
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
            ternary_quantization = True
        else:
            ternary_quantization = False

        if same_sign:
            # If J has only positive or only negative values, we can calculate the precision
            # based on the range from 0 to the maximum value.
            J_is_positive = J_min >= 0
            if J_is_positive:
                quantization_lower_bound = 0
            else:
                quantization_lower_bound = - (2 ** original_required_int_precision - 1)
        else:
            # If J has both positive and negative values, we need to consider the range
            # from the minimum to the maximum value.
            assert quantization_precision > 1, \
            f"Quantization precision {quantization_precision} must be greater than 1-bit for signed data."
            quantization_lower_bound = - (2 ** (original_required_int_precision - 1))

        if ternary_quantization:
            # Ternary quantization is treated the same as 2-bit quantization
            step_size = 2 ** (original_required_int_precision - 2)
        else:
            step_size = 2 ** (original_required_int_precision - quantization_precision)
        nonzero_mask = J != 0
        quantized_J = copy.deepcopy(J)
        quantized_J[nonzero_mask] = np.round((J[nonzero_mask] - quantization_lower_bound) / step_size) \
            * step_size + quantization_lower_bound
        if ternary_quantization and not same_sign:
            # Convert all the most negative values to the second most negative values
            # This is to ensure that we have only one unique negative value
            quantized_J[quantized_J == quantization_lower_bound] = quantization_lower_bound + step_size

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
