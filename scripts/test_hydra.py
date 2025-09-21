import hydra

@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg):
    print(cfg.traindata)
    print(cfg.testdata)

if __name__ == "__main__":
    main()