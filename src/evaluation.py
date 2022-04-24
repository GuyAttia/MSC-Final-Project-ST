# Abstract evaluator class, each model should implement its own `process_batch` and `prepare_evaluation_dataframe`
class AbstractModelEvaluator:
    def __init__(self):
        self.reset()
        
    def evaluate(self, model, dataloader, device, cutoff=4):
        self.reset()
        with torch.no_grad():
            for batch in tqdm(iter(dataloader)):
                batch = batch[0].to(device)
                self.process_batch(model, batch)

        eval_df = self.prepare_evaluation_dataframe()
        
        return eval_df
    
    def reset(self):
        pass
    
    def process_batch(self, model, batch, device):
        raise NotImplementedError('Subclasses must implement process_batch')
        
    def prepare_evaluation_dataframe(self):
        raise NotImplementedError('Subclasses must implement prepare_evaluation_dataframe')
        
class GMFEvaluator(AbstractModelEvaluator):
    def reset(self):
        self.y = []
        self.y_pred = []
        self.all_genes = []
        self.all_spots = []
        
    def process_batch(self, model, batch):
        gene_indices, spot_indices, counts = batch[:, 0], batch[:, 1], batch[:, 2]
        self.y.append(counts)
        self.all_genes.append(gene_indices)
        self.all_spots.append(spot_indices)
        gene_indices = gene_indices.to(device)
        spot_indices = spot_indices.to(device)
        counts = counts.to(device, dtype=torch.float32)
        pred = model(gene_indices, spot_indices)
        self.y_pred.append(pred.detach().cpu().numpy())
        
    def prepare_evaluation_dataframe(self, clip=True):
        y = np.concatenate(self.y)
        y_pred = np.concatenate(self.y_pred)
        if clip:
            y_pred = np.clip(y_pred, 0, np.inf)
        all_genes = np.concatenate(self.all_genes)
        all_spots = np.concatenate(self.all_spots)
        
        return pd.DataFrame({
            'gene': all_genes,
            'spot': all_spots,
            'count': y,
            'pred_count': y_pred
        })

