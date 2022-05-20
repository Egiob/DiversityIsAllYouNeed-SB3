    # train the sil model...
    def train_sil_model(self):
        for n in range(self.args.n_update):
            obs, actions, returns, weights, idxes = self.sample_batch(self.args.batch_size)
            mean_adv, num_valid_samples = 0, 0

            if obs is not None:
                # need to get the masks
                # get basic information of network..
                obs = torch.tensor(obs, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
                returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
                max_nlogp = torch.tensor(np.ones((len(idxes), 1)) * self.args.max_nlogp, dtype=torch.float32)
                if self.args.cuda:
                    obs = obs.cuda()
                    actions = actions.cuda()
                    returns = returns.cuda()
                    weights = weights.cuda()
                    max_nlogp = max_nlogp.cuda()
                # start to next...
                value, pi = self.network(obs)
                action_log_probs, dist_entropy = evaluate_actions_sil(pi, actions)
                action_log_probs = -action_log_probs
                clipped_nlogp = torch.min(action_log_probs, max_nlogp)
                # process returns
                advantages = returns - value
                advantages = advantages.detach()
                masks = (advantages.cpu().numpy() > 0).astype(np.float32)
                # get the num of vaild samples
                num_valid_samples = np.sum(masks)
                num_samples = np.max([num_valid_samples, self.args.mini_batch_size])
                # process the mask
                masks = torch.tensor(masks, dtype=torch.float32)
                if self.args.cuda:
                    masks = masks.cuda()
                # clip the advantages...
                clipped_advantages = torch.clamp(advantages, 0, self.args.clip)
                mean_adv = torch.sum(clipped_advantages) / num_samples 
                mean_adv = mean_adv.item() 
                # start to get the action loss...
                action_loss = torch.sum(clipped_advantages * weights * clipped_nlogp) / num_samples
                entropy_reg = torch.sum(weights * dist_entropy * masks) / num_samples
                policy_loss = action_loss - entropy_reg * self.args.entropy_coef

                # start to process the value loss..
                # get the value loss
                delta = torch.clamp(value - returns, -self.args.clip, 0) * masks
                delta = delta.detach()
                value_loss = torch.sum(weights * value * delta) / num_samples
                total_loss = policy_loss + 0.5 * self.args.w_value * value_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

        return mean_adv, num_valid_samples

    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.args.sil_beta)
        else:
            return None, None, None, None, None
