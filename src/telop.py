from common import CONST_SPEAKER

from essential_graphics import EssentialGraphics

from lxml.etree import parse
from lxml.etree import Element

class Telop:
    def __init__(self, text, speaker_id, starttime, endtime) -> None:
        self.text = text
        self.speaker_id = speaker_id
        self.essential_graphics = EssentialGraphics(text, starttime, endtime)

        if speaker_id == CONST_SPEAKER['horimoto'] :
            basicmotion_tree = parse('animation/telop_left_basicmotion.xml')
            opacity_tree = parse('animation/telop_left_opacity.xml')
        else :
            basicmotion_tree = parse('animation/telop_right_basicmotion.xml')
            opacity_tree = parse('animation/telop_right_opacity.xml')

        basicmotion_tag = basicmotion_tree.getroot()
        opacity_tag = opacity_tree.getroot()

        self.essential_graphics.clipitem_tag.insert(15, opacity_tag)
        self.essential_graphics.clipitem_tag.insert(15, basicmotion_tag)

telop = Telop('こんにちは', 1, 44444, 55555)
# with open('sample_telop.xml', 'w') as f:
#     f.write(telop.essential_graphics.clipitem_tag)
