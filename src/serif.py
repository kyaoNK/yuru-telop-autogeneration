from common import CONST_SPEAKER

from essential_graphics import EssentialGraphics

from lxml.etree import parse
from lxml.etree import Element

class Telop:
    def __init__(self, text, speaker_id, starttime, endtime) -> None:
        self.text = text
        self.speaker_id = speaker_id
        self.essentialgraphics = EssentialGraphics(text, starttime, endtime)

        basicmotion_tree = parse('animation/serif_basicmotion.xml')
        opacity_tree = parse('animation/serif_opacity.xml')

        basicmotion_tag = basicmotion_tree.getroot()
        opacity_tag = opacity_tree.getroot()

        self.essentialgraphics.clipitem_tag.insert(15, opacity_tag)
        self.essentialgraphics.clipitem_tag.insert(15, basicmotion_tag)

telop = Telop('こんにちは', 1, 44444, 55555)
