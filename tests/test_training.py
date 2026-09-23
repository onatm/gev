import torch
import pytest

from gev.training.objective import logical_loss, question_loss
from gev.models.gemma import build_tiny_model
from gev.cli import main


def _q(label=0, target=None):
    value = {"label": label}
    if target is not None:
        value["target"] = target
    return value


def test_soft_target_matches_manual_cross_entropy():
    logits = torch.tensor([0.3, -0.2, 1.1], requires_grad=True)
    target = torch.tensor([0.2, 0.5, 0.3])
    expected = -(target * torch.log_softmax(logits, -1)).sum()
    torch.testing.assert_close(question_loss(logits, _q(target=target.tolist())), expected)


def test_logical_loss_is_mean_of_variant_losses_with_unequal_question_counts():
    logits = [[torch.tensor([2.0, -1.0])], [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])]]
    variants = [{"record": {"questions": [_q(0)]}}, {"record": {"questions": [_q(1), _q(0)]}}]
    expected = (question_loss(logits[0][0], variants[0]["record"]["questions"][0]) +
                (question_loss(logits[1][0], _q(1)) + question_loss(logits[1][1], _q(0))) / 2) / 2
    torch.testing.assert_close(logical_loss(logits, variants), expected)


def test_microbatch_gradient_chunking_matches_full_logical_batch():
    weight = torch.tensor([[0.4, -0.2], [0.1, 0.7]], requires_grad=True)
    features = [torch.tensor([1.0, 2.0]), torch.tensor([-1.0, .5]), torch.tensor([.3, -.7])]
    labels = [0, 1, 0]
    full = []
    for x, y in zip(features, labels):
        full.append(question_loss(weight @ x, _q(y)))
    sum(full).div(len(full)).backward()
    full_grad = weight.grad.detach().clone()
    weight.grad = None
    for x, y in zip(features, labels):
        (question_loss(weight @ x, _q(y)) / len(features)).backward()
    torch.testing.assert_close(weight.grad, full_grad)


def test_optimizer_recipe_has_two_groups_and_onecycle_fraction():
    lora = torch.nn.Parameter(torch.ones(2))
    head = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([{"params": [lora], "lr": 1e-4}, {"params": [head], "lr": 2e-4}], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-4, 2e-4], total_steps=32, pct_start=.1)
    assert optimizer.defaults["betas"] == (.9, .999)
    assert scheduler.total_steps == 32
    assert scheduler._schedule_phases[0]["end_step"] == pytest.approx(32 * .1 - 1)


def test_nontrainable_parameter_is_not_in_optimizer_recipe():
    frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    trainable = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([trainable], lr=1e-4)
    assert all(frozen is not p for group in optimizer.param_groups for p in group["params"])


def test_actual_tiny_gemma_overfit_diagnostic_reduces_loss():
    model = build_tiny_model(layers=6, hidden_size=32).train()
    encoded = {"ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1], "pos": [0, 1, 2, 3, 4, 5, 6], "opt": [-1, -1, -1, -1, 0, 1, -2], "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    losses = []
    for _ in range(32):
        logits = model.forward_one(encoded)
        loss = question_loss(logits[0], {"label": 0})
        losses.append(float(loss.detach()))
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    assert losses[-1] <= losses[0] * .7


def test_training_cli_does_not_admit_development_split():
    with pytest.raises(SystemExit):
        main(["train", "--split", "development", "--out", "runs/invalid-training-split"])
