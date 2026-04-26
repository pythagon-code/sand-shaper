from typing import Callable
from ursina import Button
from ursina import Slider


def create_expand_button(toggle_callback: Callable[[], None]) -> Button:
    button = Button(text = "Expand", scale = (0.15, 0.06), position = (-0.78, 0.45))
    button.on_click = toggle_callback
    return button


def create_grid_size_slider(change_callback: Callable[[], None]) -> Slider:
    slider = Slider(min = 64, max = 512, default = 64, step = 1, x = -0.55, y = 0.45, scale = 0.6)
    slider.on_value_changed = change_callback
    return slider
