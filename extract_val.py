from common import utils
from mitos_extract_anotations import candidateSelection as cs
from common.Params import Params as P


if __name__ == '__main__':
    filter = ['*.bmp', '*.png', '*.jpg']
    file_list = utils.listFiles(P().basedir + 'normalizado/testHeStain', filter)
    params = cs.Candidates_extractor_params(file_list)
    params.candidates_json_save_path = P().basedir + 'anotations/test_cand.json'
    params.save_candidates_dir_path = P().basedir + 'test/no-mitosis/'
    params.save_mitosis_dir_path = P().basedir + 'test/mitosis/'
    params.bsave_img_keypoints = True
    params.bappend_mitosis_to_json = True

    cutter = cs.Candidates_extractor(params)
    cutter.extract()