# Night 2 continuation contract

The continuation is the frozen 1,425-record Night 2 artifact followed by a
deterministic 2,000-record replay sample from decision-v7 train.  The replay
uses `random.Random("replay:<training seed>")`; `night2-20260920` is only the
artifact-generation seed.  The prepared order and both source hashes are part
of the manifest, and the source files are revalidated whenever a prepared
directory is loaded.

The historical replay-first artifact is invalid and must not be reused.  A
stale recipe or source hash is rejected; prepare a new recipe-addressed output
instead.  Token auditing includes augmentation variants for epoch zero and
counts `state_length + question_length` against the branch cap.

An initializer is diagnostic unless its verified metadata identifies the
complete official v7 run (3,144 logical steps and 25,152 processed records,
with the official train and manifest hashes).  Continuation always loads only
weights and starts a fresh optimizer at step zero; optimizer/resume state is
never inherited.
