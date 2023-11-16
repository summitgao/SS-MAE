import data.dataset as Dataset

def get_dataset(args):
    train_dataset = args.dataset
    Dataset.set_random_seed(0)

    if train_dataset == "Houston2018":

        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = Dataset.getHSData(
            datasetType="Houston2018",
            channels=args.pca_num,
            windowSize=args.crop_size,
            batch_size=args.batch_size,
            num_workers=0,args=args)

    elif train_dataset == "Berlin":

        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = Dataset.getHSData(
            datasetType="Berlin",
            channels=args.pca_num,
            windowSize=args.crop_size,
            batch_size=args.batch_size,
            num_workers=0,args=args)

    elif train_dataset == "Augsburg":

        pretrain_loader, train_loader, test_loader, trntst_loader, all_loader = Dataset.getHSData(
            datasetType="Augsburg",
            channels=args.pca_num,
            windowSize=args.crop_size,
            batch_size=args.batch_size,
            num_workers=0,args=args)

    print("completed!")
    
    return pretrain_loader, train_loader, test_loader, trntst_loader, all_loader