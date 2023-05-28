import os, sys
import enum

class DatasetType(enum.Enum):
    Flickr30k = "flickr30k"
    PartiPrompts = "parti-prompts"
    PartiPromptsSmall = "parti-prompts-small"
    PartiPromptsMedium = "parti-prompts-medium"
    Flickr30kSmall = "flickr30k-small"
    CocoCaptionSmall = "cococaption-small"
    CocoCaptionMedium = "cococaption-medium"
    CocoCaptionLarge = "cococaption-large"