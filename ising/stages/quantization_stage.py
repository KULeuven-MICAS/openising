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

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        config: Any,
        ising_model: IsingModel | None = None,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.ising_model = ising_model
        if hasattr(self.config, "visualization"):
            self.visualization = self.config.visualization
        else:
            self.visualization = False
        self.original_quantization = self.config.quantization_precision

    def run(self) -> Any:
        """! Quantize the J of the Ising model."""

        # Get original integer precision of J and h
        original_int_j_precision, j_is_unsigned = self.calc_original_precision(self.ising_model.J)
        original_int_h_precision, h_is_unsigned = self.calc_original_precision(self.ising_model.h)
        LOGGER.info(
            f"Original required int precision for J: {original_int_j_precision}, for h: {original_int_h_precision}"
        )
        LOGGER.info(f"J is unsigned: {j_is_unsigned}, h is unsigned: {h_is_unsigned}")

        # Calculate actual original precision as maximum precision of h and J
        if original_int_h_precision == 0:
            original_precision = original_int_j_precision
        else:
            original_precision = max(original_int_j_precision, original_int_h_precision)

        if self.config.quantization:
            if self.config.combine_nodes:
                # Each node will be split into a number of replica nodes. Some different steps need to be undertaken.
                # The values will be scaled to their respective integers and the quantization precision is recalculated.
                LOGGER.info("Combine nodes is enabled, setting scale_to_integer to True.")
                self.config.scale_to_integer = True
                quantization_precision = int(
                    np.ceil(np.log2((2 ** (self.config.quantization_precision - 1) - 1) * self.config.nodes_scaling**2))
                    + 1
                )
                # Calculate the maximum allowed value to which can be quantized due to slightly smaller range
                max_quant_valJ = (2 ** (self.config.quantization_precision - 1) - 1) * self.config.nodes_scaling**2
                max_quant_valh = (2 ** (self.config.quantization_precision - 1) - 1) * self.config.nodes_scaling
            else:
                # Combine nodes is not activated
                quantization_precision = self.config.quantization_precision
                max_quant_valJ = 2 ** (quantization_precision - 1) - 1
                max_quant_valh = max_quant_valJ
            scale_to_integer = self.config.scale_to_integer if hasattr(self.config, "scale_to_integer") else False
            original_J = self.ising_model.J
            original_h = self.ising_model.h

            # Calculate the scale factor of h and map h to new range
            if self.config.scale_h and np.max(np.abs(original_h)) != 0:
                h_scale_factor_real = np.max(np.abs(original_h)) / np.max(np.abs(original_J))
                h_scale_factor = int(np.round(h_scale_factor_real))
            else:
                # Don't scale h after quantization
                h_scale_factor_real = 1
                h_scale_factor = 1
            original_h = original_h / h_scale_factor
            LOGGER.info(f"original h scaling factor is {h_scale_factor_real}, rounded scale factor is {h_scale_factor}")

            # Quantize J and h.
            quantized_matrix = self.quantize_matrix(
                matrix=original_J,
                original_precision=original_int_j_precision,
                quantization_precision=quantization_precision,
                scale_to_integer=scale_to_integer,
                max_quant_val=max_quant_valJ,
            )
            if np.max(np.abs(original_h)) != 0:
                quantized_h = self.quantize_matrix(
                    matrix=original_h,
                    original_precision=original_int_h_precision,
                    quantization_precision=quantization_precision,
                    scale=h_scale_factor,
                    scale_to_integer=scale_to_integer,
                    max_quant_val=max_quant_valh,
                )
            else:
                # bias is all zeros, don't need to quantize this.
                quantized_h = original_h
            self.config.h_scale_factor = h_scale_factor
            LOGGER.info(f"Quantization is enabled with precision: {quantization_precision}-bit.")

            quantized_model = IsingModel(
                J=np.triu(quantized_matrix, k=1),
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
            if self.config.quantization:
                ans.h_scale_factor = h_scale_factor
            else:
                ans.h_scale_factor = None
            for solver in ans.config.solvers:
                for energy_id in range(len(ans.energies[solver])):
                    ans.energies[solver][energy_id] = self.ising_model.evaluate(
                        ans.states[solver][energy_id].astype(np.float32)
                    )
                    if hasattr(ans, "tsp_energies"):
                        if ans.tsp_energies[solver][energy_id] == math.inf:
                            ans.tsp_energies[solver][energy_id] = math.inf
                        else:
                            ans.tsp_energies[solver][energy_id] = ans.energies[solver][energy_id]
            yield ans, debug_info

    @staticmethod
    def calc_original_precision(J: np.ndarray) -> tuple[int:, bool]:
        """! Calculate the original precision of the matrix.

        @type J: np.ndarray
        @param J: the input matrix
        @rtype: tuple[int, bool]
        @return: the original required int precision and whether the matrix is unsigned.
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
            original_precision = math.ceil(math.log2(max(abs(J_max), np.abs(J_min))) + 1)

        # signed or unsigned representation
        is_unsigned = same_sign

        return original_precision, is_unsigned

    # @staticmethod
    def quantize_matrix(
        self,
        matrix: np.ndarray,
        original_precision: int,
        max_quant_val: int,
        quantization_precision: int | float = 2,
        scale: float = 1.0,
        scale_to_integer: bool = False,
    ) -> np.ndarray:
        """! Quantizes a matrix to a given precision.

        @type matrix: np.ndarray
        @param matrix: the input matrix
        @type original_precision: int
        @param original_precision: the original precision of the matrix
        @type quantization_precision: int or float
        @param quantization_precision: the precision for quantization
        @type scale: float
        @param scale: the scaling factor for the matrix
        @type scale_to_integer: bool
        @param scale_to_integer: scale the values to the ones supported on chip
        @rtype: np.ndarray
        @return: a quantized matrix
        """
        J_min = np.min(matrix)
        J_max = np.max(matrix)

        if (J_min >= 0 and J_max >= 0) or (J_min <= 0 and J_max <= 0):
            same_sign = True
        else:
            same_sign = False

        LOGGER.debug(
            "Original int precision: %s, current quant: %s",
            original_precision,
            quantization_precision,
        )
        assert quantization_precision <= original_precision, (
            f"Quantization precision {quantization_precision} is larger "
            f"than the original precision {original_precision}."
        )
        assert quantization_precision == 1.5 or isinstance(quantization_precision, int)

        if quantization_precision == 1.5:
            ternary_quantization = True
        else:
            ternary_quantization = False

        # Get lower bound of the original range
        if same_sign:
            # If J has only positive or only negative values, we can calculate the precision
            # based on the range from 0 to the maximum value.
            J_is_positive = J_min >= 0
            if J_is_positive:
                quantization_lower_bound = 0
            else:
                quantization_lower_bound = -max(np.abs(J_min), np.abs(J_max))
        else:
            # If J has both positive and negative values, we need to consider the range
            # from the minimum to the maximum value.
            assert quantization_precision > 1, (
                f"Quantization precision {quantization_precision} must be greater than 1-bit for signed data."
            )
            quantization_lower_bound = -max(np.abs(J_min), np.abs(J_max))

        # Get the amount of steps that can be taken with current quantization precision.
        if ternary_quantization:
            step_num: int = 2
        else:
            step_num: int = (2**quantization_precision) - 1
            if not same_sign:
                step_num -= 1  # dismiss the most negative value

        quantized_matrix = copy.deepcopy(matrix.astype(np.float64))

        # Now calculate the step size based on whether the nodes will be split.
        if 2 ** (quantization_precision - 1) - 1 > max_quant_val and self.config.combine_nodes:
            step_size = max(np.abs(J_max), np.abs(J_min)) / max_quant_val
        else:
            step_size = 2 * max(np.abs(J_max), np.abs(J_min)) / step_num

        # Add half a step size to the lower bound to minimize the quantization error.
        quantization_lower_bound = quantization_lower_bound + step_size / 2.0
        quantization_upper_bound = np.abs(quantization_lower_bound)
        step_size = quantization_upper_bound/step_num*2
        nonzero_mask = (matrix != 0) & (matrix > quantization_lower_bound) & (matrix < quantization_upper_bound)

        quantized_matrix[nonzero_mask] = (
            np.round((matrix[nonzero_mask] - quantization_lower_bound) / step_size) * step_size
            + quantization_lower_bound
        )
        # Set values close to zero to zero due to round-off
        quantized_matrix[nonzero_mask] = np.where(
            np.isclose(quantized_matrix[nonzero_mask], 0, atol=1e-3), 0, quantized_matrix[nonzero_mask]
        )

        # Due to round-off, set values close to max or min value to respective value
        quantized_matrix[quantized_matrix <= quantization_lower_bound] = quantization_lower_bound
        quantized_matrix[quantized_matrix >= quantization_upper_bound] = quantization_upper_bound

        quantized_matrix = quantized_matrix.astype(np.float32)

        if scale_to_integer and step_size != 0:
            quantized_matrix = np.round(quantized_matrix / step_size)

        assert len(np.unique(quantized_matrix)) <= (step_num + 1), (
            f"Quantized J matrix has {len(np.unique(quantized_matrix))} unique values, "
            f"which exceeds the limit for "
            f"{quantization_precision}-bit quantization."
        )
        quantized_matrix *= scale

        LOGGER.info(
            f"Quantization details for {'J matrix' if len(matrix.shape) == 2 else 'h vector'}: original min value {
                J_min
            }, quantized min value {np.min(quantized_matrix)}, original max value {J_max}, quantized max value {
                np.max(quantized_matrix)
            }, scale factor {scale}"
        )

        return quantized_matrix

    @staticmethod
    def plot_ndarray_in_matrix(
        mat: np.ndarray,
        output: str = "vdarray_matrix.png",
        zero_as_white: bool = True,
    ):
        """! Visualize 2D ndarray in matrix
        @type mat: np.ndarray
        @param mat: input 2D ndarray
        @type output: str
        @param output: output file name
        @type zero_as_white: bool
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
            cmap.set_bad(color="white")
        else:
            mat_copy = mat

        sns.heatmap(
            mat_copy,
            ax=ax,
            cmap="viridis",  # Yellow-Orange-Red colormap
            cbar_kws={"label": "Value"},
        )

        # Add annotation to the heatmap
        # for i in range(mat_copy.shape[0]):
        #     for j in range(mat_copy.shape[1]):
        #         if not np.isnan(mat_copy[i, j]):
        #             ax.text(j + 0.5, i + 0.5, int(round(mat_copy[i, j], 0)),
        #                     ha="center", va="center", color="black", fontsize=6)

        # Add colorbar legend
        ax.figure.axes[-1].yaxis.label.set_size(12)
        ax.set_title(
            f"Shape: {mat.shape}, value min: {round(np.min(mat), 2):.2f}, max: {round(np.max(mat), 2):.2f},"
            f"mean: {round(np.mean(mat), 2):.2f}, unique levels: {len(np.unique(mat)):.2f}\n"
            f"abs value min: {round(np.min(np.abs(mat)), 2):.2f}, max: {round(np.max(np.abs(mat)), 2):.2f},"
            f"mean: {round(np.mean(np.abs(mat)), 2):.2f}, nz_density: {nz_density:.0%}",
            loc="left",
            pad=10,
            weight="bold",
            fontsize=8,
        )
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
        """! Plot the distribution of a ndarray
        @type data: np.ndarray
        @param data: input ndarray
        @type output: str
        @param output: output file name
        @type bins: int
        @param bins: number of bins for the histogram
        """

        plt.figure(figsize=(8, 6))
        plt.hist(data.reshape(-1, 1), bins=bins, alpha=0.7, color="blue", edgecolor="black")
        plt.title(
            f"Distribution of values (min: {round(np.min(data), 2):.2f}, "
            f"max: {round(np.max(data), 2):.2f}, mean: {round(np.mean(data), 2):.2f}, "
            f"unique levels: {len(np.unique(data))})",
            fontsize=10,
            weight="bold",
        )
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis="y", alpha=0.75)
        plt.tight_layout()
        plt.savefig(output, bbox_inches="tight")
        plt.close()
