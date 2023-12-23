def print_qvalues(opponent, state):
        """
        Printing the q-values after the opponent has made a move.
        This is useful for debugging, and it is also interesting to see how the q-values change during training.
        """

        q_values = opponent(state).detach().numpy() # A list containing the q-values for each action
        column_names = [f"Column{i}" for i in range(1, len(q_values) + 1)] # Create a list of column names

        # Decide the width of each column. We take the max length between each header and its corresponding value
        col_widths = [max(len(str(col)), len(f'{val:.5f}')) for col, val in zip(column_names, q_values)] # We are specifying that the q-values should be printed with 8 decimals.

        header = " | ".join([str(col).ljust(width) for col, width in zip(column_names, col_widths)]) # Column names
        values_str = " | ".join([colorize(val, q_values).ljust(width) for val, width in zip(q_values, col_widths)]) # Colorized q-values

        print(header)
        print("-" * len(header))  # Separator
        print(values_str)


def colorize(qvalue, q_values):
        """
        Returns a colorized string representing the q-value.
        The color intenisty is relative to the min and max q-value for the given state.
        Red means negative, green means positive.
        """
        # Map the value to a color intensity between 0 and 4.
        m = min(q_values); M = max(q_values)
        

        ### Code for finding the color intensity. ###
        # Intensity is relative to the min and max qvalue for the given state.
        intensity = -1

        if qvalue < 0:
            ranges = [abs(m) / 5 * i for i in range(1, 5)]
            for i, val in enumerate(ranges):
                if abs(qvalue) < val:
                    intensity = i
                    break
        else:
            ranges = [M / 5 * i for i in range(1, 5)]
            for i, val in enumerate(ranges):
                if qvalue < val:
                    intensity = i
                    break
        
        intensity = 4 if intensity == -1 else intensity # If the value is in the last range, set intensity to 4.
        

        # Define shades of green and red (these are ANSI color codes)
        shades_of_green = [226, 190, 154, 118, 82]; shades_of_red = [52, 88, 124, 160, 196]
        color_code = shades_of_green[intensity] if qvalue > 0 else shades_of_red[intensity]
        
        return f"\033[38;5;{color_code}m{qvalue:.5f}\033[0m" # Return the colorized value