from lxml.etree import Element

import re

class ClipIDManager:
    def __init__(self, root: Element) -> None:
        video_clipitem_tags = root.findall('sequence/media/video/track/clipitem')
        audio_clipitem_tags = root.findall('sequence/media/audio/track/clipitem')

        self.clipitemid_count = 1
        self.masterclipid_count = 1
        self.fileid_count = 1

        self.clipitemid_list = list()
        self.masterclipid_list = list()
        self.fileid_list = list()

        for tag in video_clipitem_tags :
            self.clipitemid_list.append(int(re.sub(r'\D', '', tag.attrib['id'])))
            masterclipid = int(re.sub(r'\D', '', tag.find('masterclipid').text))
            if masterclipid not in self.masterclipid_list:
                self.masterclipid_list.append(masterclipid)
            fileid = int(re.sub(r'\D', '', tag.find('file').attrib['id']))
            if fileid not in self.fileid_list:
                self.fileid_list.append(fileid)

        for tag in audio_clipitem_tags :
            self.clipitemid_list.append(int(re.sub(r'\D', '', tag.attrib['id'])))
            masterclipid = int(re.sub(r'\D', '', tag.find('masterclipid').text))
            if masterclipid not in self.masterclipid_list:
                self.masterclipid_list.append(masterclipid)
            fileid = int(re.sub(r'\D', '', tag.find('file').attrib['id']))
            if fileid not in self.fileid_list:
                self.fileid_list.append(fileid)

    def get_new_clipitemid(self) -> int:
        while True :
            if self.clipitemid_count in self.clipitemid_list :
                self.clipitemid_count += 1
            else :
                self.clipitemid_list.append(self.clipitemid_count)
                return self.clipitemid_count

    def get_new_masterclipid(self) -> int:
        while True :
            if self.masterclipid_count in self.masterclipid_list:
                self.masterclipid_count += 1
            else :
                self.masterclipid_list.append(self.masterclipid_count)
                return self.masterclipid_count
            
    def get_new_fileid(self) -> int:
        while True :
            if self.fileid_count in self.fileid_list:
                self.fileid_count += 1
            else :
                self.fileid_list.append(self.fileid_count)
                return self.fileid_count
    
    def print(self) -> None:
        print(f'count')
        print(f'\tclipitemid:   {self.clipitemid_count}')
        print(f'\tmasterclipid: {self.masterclipid_count}')
        print(f'\tfileid:       {self.fileid_count}')
        print(f'id list')
        print(f'\tclipitemid:   {self.clipitemid_list}')
        print(f'\tmasterclipid: {self.masterclipid_list}')
        print(f'\tfileid:       {self.fileid_list}')
        