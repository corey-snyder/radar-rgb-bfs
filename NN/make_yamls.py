import yaml
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-template", help="path of template yaml file. makes copies with different seeds", type=str)
    parser.add_argument("-dir", help="path of new yaml files", type=str)
    parser.add_argument("-n", help="number of yamls", type=int)
    args = parser.parse_args()
    yaml_template = args.template
    out_dir = args.dir
    n = args.n

    with open(yaml_template) as f:
        list_doc = yaml.load(f, Loader=yaml.FullLoader)
    
    for model_type in ['in','before','after']:
        for ii in range(n):
            list_doc['seed'] = ii
            list_doc['radar_inclusion_type'] = model_type
            print(out_dir + "setup_" + model_type + '_' + str(ii) + ".yaml")
            with open(out_dir + "setup_" + model_type + '_' + str(ii) + ".yaml", "w+") as f:
                yaml.dump(list_doc, f)
