from ising.stages import LOGGER
from typing import Any
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
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
        self.ising_model = ising_model
        if hasattr(self.config, "visualization"):
            self.visualization = self.config.visualization
        else:
            self.visualization = False

    def run(self) -> Any:
        """! Quantize the J of the Ising model."""
        original_int_j_precision, j_is_unsigned = self.calc_original_precision(self.ising_model.J)
        original_int_h_precision, h_is_unsigned = self.calc_original_precision(self.ising_model.h)
        LOGGER.info(f"Original required int precision for J: {original_int_j_precision}, "
                    f"for h: {original_int_h_precision}")
        LOGGER.info(f"J is unsigned: {j_is_unsigned}, h is unsigned: {h_is_unsigned}")

        if original_int_h_precision == 0:
            original_precision = original_int_j_precision
        else:
            original_precision = max(original_int_j_precision, original_int_h_precision)

        if self.config.quantization:
            quantization_precision = self.config.quantization_precision
            original_J = self.ising_model.J
            quantized_J = self.quantize_matrix(J=original_J,
                                               original_precision=original_int_j_precision,
                                               quantization_precision=quantization_precision)
            quantized_h = self.quantize_matrix(J=self.ising_model.h,
                                               original_precision=original_int_h_precision,
                                               quantization_precision=quantization_precision)
            LOGGER.info(f"Quantization is enabled with precision: {quantization_precision}-bit.")

            quantized_model = IsingModel(
                J=quantized_J,
                h=quantized_h,
                c=self.ising_model.c,
            )
        else:
            LOGGER.debug("Quantization is disabled.")
            quantized_model = copy.deepcopy(self.ising_model)

        if self.visualization:
            self.plot_ndarray_in_matrix(quantized_model.J + quantized_model.J.T, output="J_matrix.png")
            self.plot_ndarray_in_matrix(quantized_model.h.reshape(-1, 1), output="h_matrix.png")
            self.plot_ndarray_distribution(quantized_model.J + quantized_model.J.T, output="J_distribution.png")
            self.plot_ndarray_distribution(quantized_model.h, output="h_distribution.png")

        self.kwargs["config"] = self.config
        self.kwargs["ising_model"] = quantized_model
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for ans, debug_info in sub_stage.run():
            ans.ising_model = self.ising_model
            ans.quantized_model = quantized_model
            ans.original_int_j_precision = original_int_j_precision
            ans.original_int_h_precision = original_int_h_precision
            ans.original_int_precision = original_precision
            for energy_id in range(len(ans.energies)):
                ans.energies[energy_id] = self.ising_model.evaluate(ans.states[energy_id])
                if hasattr(ans, "tsp_energies"):
                    if ans.tsp_energies[energy_id] == math.inf:
                        ans.tsp_energies[energy_id] = math.inf
                    else:
                        ans.tsp_energies[energy_id] = ans.energies[energy_id]
            yield ans, debug_info

    @staticmethod
    def calc_original_precision(J: np.ndarray) -> tuple[int:, bool]:
        """! Calculate the original precision of the matrix.

        @param J: the input matrix

        @return: the original required int precision
        @return: whether the matrix is unsigned
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
            original_precision = math.ceil(math.log2(maximum_value + 1))
        else:
            # If J has both positive and negative values, we need to consider the range
            # from the minimum to the maximum value.
            # The range is (J_max - J_min), and we add 1 to ensure we cover the full range.
            # This is because we need to represent both positive and negative values.
            original_precision = math.ceil(math.log2(abs(J_max - J_min) + 1))

        # signed or unsigned representation
        is_unsigned = same_sign

        return original_precision, is_unsigned

    @staticmethod
    def quantize_matrix(
            J: np.ndarray,
            original_precision: int,
            quantization_precision: int | float = 2
            ) -> np.ndarray:
        """! Quantizes a matrix to a given precision.

        @param J: the input matrix
        @param original_precision: the original precision of the matrix
        @param quantization_precision: the precision for quantization

        @return: a quantized matrix
        """
        J_min = int(np.min(J))
        J_max = int(np.max(J))
        if (J_min >= 0 and J_max >= 0) or (J_min <= 0 and J_max <= 0):
            same_sign = True
        else:
            same_sign = False

        LOGGER.debug(
            "Original int precision: %s, current quant: %s",
            original_precision,
            quantization_precision,
        )
        assert quantization_precision <= original_precision, \
            f"Quantization precision {quantization_precision} is larger " \
            f"than the original precision {original_precision}."
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
                quantization_lower_bound = - (2 ** original_precision - 1)
        else:
            # If J has both positive and negative values, we need to consider the range
            # from the minimum to the maximum value.
            assert quantization_precision > 1, \
            f"Quantization precision {quantization_precision} must be greater than 1-bit for signed data."
            quantization_lower_bound = - (2 ** (original_precision - 1))

        if same_sign:
            if ternary_quantization:
                step_num: int = 2
            else:
                step_num: int = 2 ** (quantization_precision - 1)
        else:
            # dismiss the most negative value to balance the pos/neg range of binary representation
            if ternary_quantization:
                step_num: int = 2
            else:
                step_num: int = (2 ** quantization_precision) - 1
                step_num = step_num - 1  # dismiss the most negative value
        step_size: float = (2 ** original_precision) / step_num

        nonzero_mask = J != 0
        quantized_J = copy.deepcopy(J)
        quantization_upper_bound = quantization_lower_bound + step_num * step_size
        quantized_J[quantized_J < quantization_lower_bound] = quantization_lower_bound
        quantized_J[quantized_J > quantization_upper_bound] = quantization_upper_bound
        quantized_J[nonzero_mask] = np.round((J[nonzero_mask] - quantization_lower_bound) / step_size) \
            * step_size + quantization_lower_bound

        assert len(np.unique(quantized_J)) <= (step_num + 1), \
            f"Quantized J matrix has {len(np.unique(quantized_J))} unique values, " \
            f"which exceeds the limit for" \
            f"{quantization_precision}-bit quantization."

        return quantized_J

    @staticmethod
    def plot_ndarray_in_matrix(
        mat: np.ndarray,
        output: str = "vdarray_matrix.png",
        zero_as_white: bool = True,
        ):
        """ ! Visualize 2D ndarray in matrix
        @param mat: input 2D ndarray
        @param output: output file name
        @param zero_as_white: whether to plot zero values as white
        """

        fig, ax = plt.subplots(1, 1)

        nz_density = np.count_nonzero(mat) / mat.size
        nz_density = round(nz_density, 2)

        if zero_as_white:
            mat_copy = mat.copy().astype(float)
            mat_copy[mat_copy == 0] = np.nan
            # Get the viridis colormap and set the color for bad values (NaN) to white
            cmap = cm.get_cmap("viridis").copy()
            cmap.set_bad(color='white')
        else:
            mat_copy = mat

        sns.heatmap(mat_copy,
                    ax=ax,
                    cmap="viridis",  # Yellow-Orange-Red colormap
                    cbar_kws={"label": "Value"})

        # Add annotation to the heatmap
        # for i in range(mat_copy.shape[0]):
        #     for j in range(mat_copy.shape[1]):
        #         if not np.isnan(mat_copy[i, j]):
        #             ax.text(j + 0.5, i + 0.5, int(round(mat_copy[i, j], 0)),
        #                     ha="center", va="center", color="black", fontsize=6)

        # Add colorbar legend
        ax.figure.axes[-1].yaxis.label.set_size(12)
        ax.set_title(
        f"Shape: {mat.shape}, value min: {round(np.min(mat), 2)}, max: {round(np.max(mat), 2)},"
        f"mean: {round(np.mean(mat), 2)}, unique levels: {len(np.unique(mat))}\n"
        f"abs value min: {round(np.min(np.abs(mat)), 2)}, max: {round(np.max(np.abs(mat)), 2)},"
        f"mean: {round(np.mean(np.abs(mat)), 2)}, nz_density: {nz_density:.0%}",
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
        plt.close()

    @staticmethod
    def plot_ndarray_distribution(
        data: np.ndarray,
        output: str = "ndarray_distribution.png",
        bins: int = 50,
        ):
        """ ! Plot the distribution of a ndarray
        @param data: input ndarray
        @param output: output file name
        @param bins: number of bins for the histogram
        """

        plt.figure(figsize=(8, 6))
        plt.hist(data.reshape(-1, 1), bins=bins, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f"Distribution of values (min: {round(np.min(data), 2)}, "
                  f"max: {round(np.max(data), 2)}, mean: {round(np.mean(data), 2)}, "
                  f"unique levels: {len(np.unique(data))})",
                  fontsize=10, weight="bold")
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(output)
        plt.close()
