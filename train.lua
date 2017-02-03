-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")
require("rnn")
require("nngraph")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'qa01.hdf5', 'data file')

-- Hyperparameters
cmd:option('-M',128,'mini-batch size')
cmd:option('-eta',0.01,'learning rate hyperparameter for lr/nn')
cmd:option('-N',250,'num epochs hyperparameter for lr/nn')
cmd:option('-D0',20,'num outputs of lookup layer of nn')
cmd:option('-Dmt',20,'dimension of m(t) in Attentive Reader model')
cmd:option('-Dg',20,'dimension of g(d,q) in Attentive Reader model')
cmd:option('-Dlstm',20,'num outputs of LSTM layer')

-- LSTM parameters
cmd:option('-rnn_type','flstm', 'type of rnn model. options are lstm / flstm / gru')
cmd:option('-mineta', 0.00001, 'minimum learning rate')
cmd:option('-saturate', 400, 'epoch at which linear decayed LR will reach minlr')
cmd:option('-stack',1,'num layers of stacked LSTM')
cmd:option('-dropout',0.68,'num layers of stacked LSTM')
cmd:option('-bidirectional', false, 'use a Bidirectional RNN/LSTM (nn.BiSequencer instead of nn.Sequencer)')
cmd:option('-save',false,'whether to save model')
cmd:option('-saveminacc',0,'minimum accuracy on test set required to save model')
cmd:option('-load','','specify model to load')
cmd:option('-cuda',false,'whether to use cuda')

function main()
   -- Parse input params
   opt = cmd:parse(arg)
   load()

   if opt.cuda then
    require("cunn")
    print("Using Cuda")
   else
    print("Using CPU")
   end

   opt.bidirectional = true
   runATR()
end

function runATR()
  trimData()
  createATR()
  trainATR()
  local accuracy = testATR()
  print('Final accuracy = ' .. accuracy)
end

function createATR()
  embed_size = opt.D0
  lstm_size = opt.Dlstm

  Yd = nn.Sequential()
  Yd:add(nn.LookupTable(nwords, embed_size)) -- [LS x batch_size x lstm_size]
  Yd:add(nn.SplitTable(1,3))

  rnns = {}
  stepmodule = nn.Sequential()
  if opt.rnn_type == 'flstm' then
    for i = 1, opt.stack do
      rnns[i] = nn.FastLSTM(embed_size, lstm_size)
    end
  elseif opt.rnn_type == 'gru' then
    for i = 1, opt.stack do
      rnns[i] = nn.GRU(embed_size, lstm_size, nil, opt.dropout / 2)
    end
  elseif opt.rnn_type == 'lstm' then
    for i = 1, opt.stack do
      rnns[i] = nn.LSTM(embed_size, lstm_size)
    end
  else
    print("Invalid rnn_type specified")
    return
  end

  for i = 1, opt.stack do
    stepmodule:add(rnns[i])
    -- only add drop out if more than 1 layers and not gru
    if opt.stack > 1 and opt.dropout > 0 and opt.rnn_type ~= 'gru' then
      stepmodule:add(nn.Dropout(opt.dropout))
    end
  end

  if opt.bidirectional then
    bilstm_size = lstm_size * 2
    local brnn = nn.BiSequencer(stepmodule:clone())
    stepmodule = nn.Sequential()
    stepmodule:add(brnn)
    Yd:add(stepmodule)
  else
    Yd:add(nn.Sequencer(stepmodule))
  end

  -- Compute u = q_forward(|Q|) + q_backward(1)
  -- Questions and Stories use separate BiLSTM but same architecture
  u = Yd:clone() -- table of size LQ : [batch_size x bilstm_size]
  u:add(nn.ConcatTable()
    :add(nn.Sequential() -- final q_forward
      :add(nn.SelectTable(-1)) -- 1d array [lstm_size]
      :add(nn.Narrow(2, 1, lstm_size))) -- [lstm_size]
    :add(nn.Sequential() -- final q_backward
      :add(nn.SelectTable(1))
      :add(nn.Narrow(2, lstm_size + 1, lstm_size))))
  u:add(nn.JoinTable(2)) -- [batch_size x bilstm_size]

  -- W_um * u
  Wum_u = nn.Sequential()
  Wum_u:add(nn.Linear(bilstm_size, opt.Dmt)) -- [batch_size x Dmt]
  Wum_u:add(nn.Replicate(LS, 1)) -- [LS x batch_size x Dmt]
  Wum_u:add(nn.Reshape(LS * opt.M, opt.Dmt)) -- [(LS * batch_size) x Dmt]

  -- input Yd, is of size LS : [batch_size x bilstm_size]
  Wym_Yd = nn.Sequential()
  Wym_Yd:add(nn.JoinTable(1)) -- [(LS * batch_size) x bilstm_size]
  Wym_Yd:add(nn.Linear(bilstm_size, opt.Dmt)) -- [(LS * batch_size) x Dmt]

  -- network to build attention score S
  M = nn.Sequential()
  M:add(nn.ParallelTable():add(Wum_u):add(Wym_Yd))
  M:add(nn.CAddTable())
  M:add(nn.Tanh())-- [(LS * batch_size) x Dmt]

  -- w_ms^T * m(t)
  S = nn.Sequential()
  S:add(nn.Linear(opt.Dmt, 1)) -- [(LS * batch_size) x 1]
  S:add(nn.Reshape(LS, opt.M)) -- [LS x batch_size]
  S:add(nn.Transpose({1,2})) -- [batch_size x LS]
  S:add(nn.SoftMax()) -- attention S = { s(t) }

  -- to compute r, uses Y_d as input to:
  -- 1. compute attention s = function of Y_d
  -- 2. compute document representation r = Y_d * s
  R = nn.Sequential()
  R:add(nn.ParallelTable()
    :add(nn.Reshape(1, LS, true)) -- S = [batch_size x 1 x LS]
    :add(nn.Sequential() -- Yd
      :add(nn.JoinTable(2)) -- [batch_size x (LS * bilstm_size)]
      :add(nn.Reshape(LS, bilstm_size, true)))) -- [batch_size x LS x bilstm_size]
  R:add(nn.MM()) -- [batch_size x 1 x bilstm_size]
  R:add(nn.Squeeze()) -- [batch_size x bilstm_size]

  G = nn.Sequential() -- G_ar(d,q)
  G:add(nn.ParallelTable()
    :add(nn.Linear(bilstm_size, opt.Dg))  -- W_ug * u = [batch_size x Dg]
    :add(nn.Linear(bilstm_size, opt.Dg))) -- W_rg * r = [batch_size x Dg]
  G:add(nn.CAddTable())
  G:add(nn.Tanh()) -- [batch_size x Dg]

  A = nn.Sequential() -- P(a | d, q)
  A:add(nn.Linear(opt.Dg, nwords)) -- [batch_size x nwords]
  A:add(nn.LogSoftMax())

  x_inp = nn.Identity()():annotate({name = 'x', description = 'memories'})
  q_inp = nn.Identity()():annotate({name = 'q', description  = 'query'})

  nng_Yd = Yd(x_inp):annotate({name = 'Yd', description = 'memory embeddings'})
  nng_u = u(q_inp):annotate({name = 'u', description = 'query embeddings'})
  nng_M = M({nng_u, nng_Yd}):annotate({name = 'M', description = 'intermediate attention'})
  nng_S = S(nng_M):annotate({name = 'S', description = 'normalized attention'})
  nng_R = R({nng_S, nng_Yd}):annotate({name = 'R', description = 'doc representation'})
  nng_G = G({nng_u, nng_R}):annotate({name = 'G', description = 'joint embedding'})
  nng_A = A(nng_G):annotate({name = 'A', description = 'final word scores'})

  model = nn.gModule({x_inp, q_inp}, {nng_A})

  -- since each batch is one long sequence we want to remeber
  -- the hidden state between runs
  if opt.rnn_type == 'gru' then
    model:remember('eval')
  else
    model:remember('both')
  end
  if opt.cuda then
    model:cuda()
    crit = nn.ClassNLLCriterion():cuda()
  else
    crit = nn.ClassNLLCriterion()
  end
  crit.sizeAverage = false
end

function trainATR()
  print(string.format(
    "Training Attentive Reader with Eta = %f, Minibatch Size = %d, " ..
    "# Epochs = %d, D0 = %d, Dmt = %d, Dg = %d",
    opt.eta, opt.M, opt.N, embed_size, opt.Dmt, opt.Dg))

  local timer = torch.Timer()
  local eta = opt.eta
  local preds
  local Xstory, Xquestion, Y
  local params, gradparams = model:getParameters()
  if opt.cuda then
    trainLoss = torch.zeros(opt.N):cuda()
    stories = train_stories:t():cuda()
    questions = train_questions:t():cuda()
    answers = train_answers:cuda()
  else
    trainLoss = torch.zeros(opt.N)
    stories = train_stories:t()
    questions = train_questions:t()
    answers = train_answers
  end

  model:zeroGradParameters()

  for i = 1, opt.N do
    idxs = torch.randperm(nq):long()
    model:training() -- make sure drop out is used in training
    for j = 1, math.ceil(nq / opt.M)  do
      model:forget() -- forget states for other stories

      last_index = j * opt.M < nq and j * opt.M or nq
      first_index = last_index - opt.M + 1
      idx = idxs:index(1, torch.range(first_index, last_index):long())

      -- transpose so that after Lookup and Split the data is in table format
      -- where each entry is a [Batch Size x Embedding] matrix
      Xstory = stories:index(2, idx)
      Xquestion = questions:index(2, idx)
      Y = answers:index(1, idx)

      preds = model:forward({Xstory, Xquestion})

      trainLoss[i] = trainLoss[i] + crit:forward(preds, Y)
      dLdpreds = crit:backward(preds, Y)

      model:zeroGradParameters()
      model:backward({Xstory, Xquestion}, dLdpreds)

      model:updateParameters(eta)
    end
    local accuracy = testATR()
    print(
      "Done with epoch"..i..". Loss = "..trainLoss[i]..". "..
      "Time = "..timer:time().real..". Accuracy = "..accuracy)

    if opt.save and accuracy >= opt.saveminacc then
      acc = torch.LongTensor({accuracy*10000}):double()[1]/100
      torch.save(opt.datafile..".atr."..acc, {model = model})
    end

    eta = eta + (opt.mineta - opt.eta)/opt.saturate
    eta = math.max(opt.mineta, eta)
    collectgarbage()
  end
end

function testATR()
  model:evaluate() -- make sure drop out is not used in testing
  nq_test = test_stories:size(1)

  if opt.cuda then
    test_s = test_stories:t():cuda()
    test_q = test_questions:t():cuda()
  else
    test_s = test_stories:t()
    test_q = test_questions:t()
  end

  local Y_hat = torch.zeros(nq_test)
  for j = 1, math.ceil(nq_test / opt.M)  do
    model:forget()

    last_index = j * opt.M < nq_test and j * opt.M or nq_test
    first_index = last_index - opt.M + 1
    idx = torch.range(first_index, last_index):long()

    xs = test_s:index(2, idx)
    xq = test_q:index(2, idx)

    local preds = model:forward({xs,xq})

    local _, y_hat = torch.max(preds:double(), 2)
    for m = 1, opt.M do
      Y_hat[idx[m]] = y_hat[m]
    end
  end

  local correct = torch.eq(Y_hat:long() - test_answers, 0):sum()
  return correct / Y_hat:size(1)
end

function makePosEncMat(input)
  input:zero()

  if input:dim() == 3 then
    num_sent , sent_len, embed_size = input:size(1), input:size(2), input:size(3)
    for i=1, num_sent do
      for j=1, sent_len do
        for k=1, embed_size do
          input[i][j][k] = (1-j/sent_len) - (k/embed_size)*(1- (2*j/sent_len))
        end
      end
    end
  else
    sent_len, embed_size = input:size(1), input:size(2)
    for j=1, sent_len do
      for k=1, embed_size do
        input[j][k] = (1-j/sent_len) - (k/embed_size)*(1- (2*j/sent_len))
      end
    end
  end
end

function writeToFile(obj,f)
  local myFile = hdf5.open(f, 'w')
  for k,v in pairs(obj) do
    myFile:write(k, v)
  end
  myFile:close()
end

function computePaddingBoundary(tensor)
  local tensor_bound = torch.zeros(tensor:size(1)):long()
  for i = 1, tensor:size(1) do
    for j = 1, tensor:size(2) do
      if tensor[i][j] == idx_pad then
        tensor_bound[i] = j - 1
        break
      end
    end
  end
  return tensor_bound
end

function trimData()
  train_stories_bound   = computePaddingBoundary(train_stories)
  train_questions_bound = computePaddingBoundary(train_questions)
  test_stories_bound    = computePaddingBoundary(test_stories)
  test_questions_bound  = computePaddingBoundary(test_questions)

  nq = train_stories:size(1)
  local max_LS_train = train_stories_bound:max()
  local max_LS_test  = test_stories_bound:max()
  LS = max_LS_train < max_LS_test and max_LS_test or max_LS_train

  local max_LQ_train = train_questions_bound:max()
  local max_LQ_test  = test_questions_bound:max()
  LQ = max_LQ_train < max_LQ_test and max_LQ_test or max_LQ_train

  if LS < 500 then
    train_stories = train_stories:sub(1,nq,1,LS)
    train_markers = train_markers:sub(1,nq,1,LS)
    test_stories = test_stories:sub(1,nq,1,LS)
    test_markers = test_markers:sub(1,nq,1,LS)
  else
    -- keep most recent memory that make up at most 300 tokens 
    LS = 300
    trunc_stories_train = torch.LongTensor(nq, LS):fill(idx_pad)
    trunc_stories_test  = torch.LongTensor(nq, LS):fill(idx_pad)
    for i = 1, nq do
      trunc_start = train_stories_bound[i] - LS + 1
      trunc_start = trunc_start < 1 and 1 or trunc_start
      for j = trunc_start, train_stories_bound[i] do
        if train_stories[i][j] == idx_start then
          trunc_length = train_stories_bound[i] - j + 1
          trunc_stories_train[i]:narrow(1, 1, trunc_length):copy(
            train_stories[i]:sub(j, train_stories_bound[i]))
          break
        end
      end
      trunc_start = test_stories_bound[i] - LS + 1
      trunc_start = trunc_start < 1 and 1 or trunc_start
      for j = trunc_start, test_stories_bound[i] do
        if test_stories[i][j] == idx_start then
          trunc_length = test_stories_bound[i] - j + 1
          trunc_stories_test[i]:narrow(1, 1, trunc_length):copy(
            test_stories[i]:sub(j, test_stories_bound[i]))
          break
        end
      end
    end
    train_stories = trunc_stories_train
    test_stories = trunc_stories_test
    print('Truncated data to '..LS..' tokens')
    print('New data size is:')
    print(string.format('Train: %d x %d', train_stories:size(1), train_stories:size(2)))
    print(string.format('Test: %d x %d', test_stories:size(1), test_stories:size(2)))
  end

  train_questions = train_questions:sub(1,nq,1,LQ)
  train_answers = train_answers:sub(1,nq)
  test_questions = test_questions:sub(1,nq,1,LQ)
  test_answers = test_answers:sub(1,nq)

  print(string.format('LS = %d, LQ = %d, nq = %d',
    LS, LQ, nq))
end

function load()
  -- get the data out of the datafile
  local f = hdf5.open(opt.datafile, 'r')
  local data = f:all()

  idx_start = data.idx_start[1]
  idx_end   = data.idx_end[1]
  idx_pad   = data.idx_pad[1]
  idx_rare  = data.idx_rare[1]

  nwords = data.nwords[1]

  train_stories   = data.train_stories:long() -- [# Questions x Max Story Length]
  train_markers   = data.train_markers:long() -- same size as stories
  train_questions = data.train_questions:long() -- [# Questions x Max Q Length]
  train_answers   = data.train_answers:long() -- [# Questions x 1]
  train_facts     = data.train_facts:long() -- [# Questions x Max Fact Length]

  test_stories   = data.test_stories:long()
  test_markers   = data.test_markers:long()
  test_questions = data.test_questions:long()
  test_answers   = data.test_answers:long()
  test_facts     = data.test_facts:long()

end


main()
