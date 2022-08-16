import os
import numpy
import torch
import collections
import random


def distance(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


class stringmatrix:
    def __init__(self, rows, cols, val=None):
        self.internal = numpy.uint8(numpy.zeros((rows, cols + 1)))
        if val is not None:
            self.internal[:] = ord(val)
        self.internal[:, -1] = ord("\n")

    def set(self, row, col, char):
        self.internal[row][col] = ord(char)

    def get(self):
        tmp = self.internal.flatten()
        tmp = list(tmp)
        tmp = [chr(i) for i in tmp]
        return "".join(tmp)


class Buffer:
    def __init__(self, size):
        self.size = size
        self.data = collections.deque(maxlen=size)

    def randomize(self):
        tmp = list(self.data)
        random.shuffle(tmp)
        self.data = collections.deque(tmp, maxlen=self.size)

    def get(self, batchsize):
        tmp = []
        for i in range(batchsize):
            tmp.append(self.data.pop())
        return tmp

    def getvector(self, batchsize, shape):
        resultingsize = (batchsize, shape)
        state, stateafter = torch.zeros(resultingsize), torch.zeros(resultingsize)
        reward, action = torch.zeros(batchsize), torch.zeros(batchsize)

        tmp = self.get(batchsize)
        for i, (s, r, a, ss) in enumerate(tmp):
            state[i] = s
            reward[i] = r
            action[i] = a
            stateafter[i] = ss

        return state, stateafter, reward, action


class DQN:
    def __init__(self, backbone):
        self.net = backbone

    def selectsingle(self, vectorstate):
        with torch.no_grad():
            vectorstate = vectorstate.unsqueeze(0)
            tmp = self.net(vectorstate)
            softmax = torch.nn.functional.softmax(tmp, dim=1)[0]
            softmax = softmax.cpu().numpy()
            obj_list = list(range(softmax.shape[0]))
            return int(numpy.random.choice(obj_list, p=softmax))

    def qabatch(self, vectorstate, action):
        tmp = self.net(vectorstate.cuda())
        out = [tmp[i][int(action[i])] for i in range(action.shape[0])]
        return torch.stack(out, dim=0)

    def qbatch(self, vectorstate):
        tmp = self.net(vectorstate.cuda())
        return torch.max(tmp, dim=1)[0]


def tuneDQN(dqn, replaybuffer, gamegenerator, parameters):
    nbtries, nbsteps, lr, alea, batchsize = parameters
    A = gamegenerator.getactionssize()
    S = gamegenerator.getvectorstatesize()

    print("exploration")
    dqn.net.cpu()
    valid = 0
    for i in range(nbtries):
        game = gamegenerator.getanother()
        for j in range(nbsteps):
            if game.isfinalstate():
                break

            s = game.getvectorstate()
            if random.randint(0, 100) <= alea:
                a = game.adhocmove()
            else:
                a = dqn.selectsingle(s)
            r = game.update(a)
            ss = game.getvectorstate()

            replaybuffer.data.append((s, r, a, ss))
            valid += 1

    print("tuning")
    replaybuffer.randomize()
    dqn.net.cuda()
    optimizer = torch.optim.Adam(dqn.net.parameters(), lr=lr)
    valid = int(valid * 0.75 / batchsize)
    valid = max(valid, len(replaybuffer.data) // 10 // batchsize)

    for i in range(valid):
        state, stateafter, reward, action = replaybuffer.getvector(batchsize, S)
        qas = dqn.qabatch(state, action)
        qss = dqn.qbatch(stateafter)
        loss = (qas - 0.95 * qss - reward.cuda()).abs().sum() / batchsize

        if i % 10 == 9:
            print(i, loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn.net.parameters(), 3)
        optimizer.step()

    print("eval")
    dqn.net.cpu()
    nbtries = nbtries // 10
    meansR = 0
    for i in range(nbtries):
        game = gamegenerator.getanother()
        R = 0
        for j in range(nbsteps):
            if game.isfinalstate():
                break

            s = game.getvectorstate()
            a = dqn.selectsingle(s)
            r = game.update(a)
            ss = game.getvectorstate()
            R += r

            replaybuffer.data.append((s, r, a, ss))
        meansR += R
    print("means R=", meansR / nbtries)


class FrozenLake:
    def __init__(self):
        self.size = 4
        self.holes = [(2, 1), (3, 1), (0, 3)]
        self.start = (random.randint(0, 100) % self.size, 0)
        self.final = (random.randint(0, 100) % self.size, 2)

        self.isvalid = True
        self.steps = [self.start]
        self.current = self.start

    def getanother(self):
        return FrozenLake()

    def isfinalstate(self):
        return not self.isvalid

    def copy(self):
        out = FrozenLake()
        out.size = self.size
        out.holes = self.holes
        out.start = self.start
        out.final = self.final

        out.isvalid = self.isvalid
        out.step = self.steps.copy()
        out.current = self.current
        return out

    # static description of the game
    def getactionssize(self):
        return 4

    def getvectorstatesize(self):
        grids = ["holes", "position", "final", "knownwithold"]
        return len(grids) * self.size * self.size

    # game instance
    def update_(self, action):
        if not self.isvalid:
            print("update an invalid state")
            quit()

        previousD = distance(self.final, self.current)

        if action == 0:
            self.current = (self.current[0] - 1, self.current[1])
        if action == 1:
            self.current = (self.current[0] + 1, self.current[1])
        if action == 2:
            self.current = (self.current[0], self.current[1] - 1)
        if action == 3:
            self.current = (self.current[0], self.current[1] + 1)

        if 0 <= self.current[0] < self.size and 0 <= self.current[1] < self.size:
            if self.current in self.holes:
                self.isvalid = False
                return -1000
            if self.current == self.final:
                self.isvalid = False
                return 1000 - len(self.steps) * 2
        else:
            self.isvalid = False
            return -1000

        self.steps.append(self.current)
        nextD = distance(self.final, self.current)
        if nextD < previousD:
            return 1
        else:
            return 0

    def update(self, action):
        # return self.update_(action) # -> holes kill

        tmp = self.copy()
        r = tmp.update_(action)
        if r < -500:
            return -2  # -> hole == wall
        else:
            return self.update_(action)

    def getvectorstate(self):
        if not self.isvalid:
            return -torch.ones(self.getvectorstatesize())

        gridshape = (self.size, self.size)
        gridsize = self.size * self.size
        holes = torch.ones(gridshape)
        for r, c in self.holes:
            holes[r][c] = 0

        position = torch.zeros(gridshape)
        position[self.current[0]][self.current[1]] = 1

        final = torch.zeros(gridshape)
        final[self.final[0]][self.final[1]] = 1

        knownwithold = holes.clone()
        for i, (r, c) in enumerate(self.steps):
            knownwithold[r][c] = (i + 1) / len(self.steps)

        out = [holes, position, final, knownwithold]
        out = [x.flatten() for x in out]
        return torch.cat(out, dim=0)

    def display(self):
        if not self.isvalid:
            return "GAME OVER"
        tmp = stringmatrix(self.size, self.size, "V")
        for r, c in self.steps:
            tmp.set(r, c, "X")
        tmp.set(self.start[0], self.start[1], "S")
        tmp.set(self.final[0], self.final[1], "F")
        tmp.set(self.current[0], self.current[1], "C")
        for r, c in self.holes:
            tmp.set(r, c, "H")
        return tmp.get()

    # heuristic
    def adhocmove(self):
        if not self.isvalid:
            print("adhoc move on invalid")
            quit()

        candidate = [0, 1, 2, 3]
        possible = []
        for i in candidate:
            other = self.copy()
            tmp = other.update(i)
            if tmp > -500 and other.current != self.current:
                possible.append(i)
        return possible[random.randint(0, 100) % len(possible)]


if __name__ == "__main__":
    generator = FrozenLake()
    game = generator.getanother()
    R = 0
    for i in range(50):
        if not game.isvalid:
            break
        print("=====")
        print(game.display(), R)
        R += game.update(game.adhocmove())
    print(game.display(), R)

    replaybuffer = Buffer(size=500000)
    vectorshape = generator.getvectorstatesize()
    actionsize = generator.getactionssize()

    backbone = torch.nn.Sequential()
    backbone.add_module("01", torch.nn.Linear(vectorshape, 512))
    backbone.add_module("02", torch.nn.LeakyReLU())
    backbone.add_module("03", torch.nn.Linear(512, 512))
    backbone.add_module("04", torch.nn.LeakyReLU())
    backbone.add_module("05", torch.nn.Linear(512, 512))
    backbone.add_module("06", torch.nn.LeakyReLU())
    backbone.add_module("07", torch.nn.Linear(512, 512))
    backbone.add_module("08", torch.nn.LeakyReLU())
    backbone.add_module("09", torch.nn.Linear(512, 512))
    backbone.add_module("10", torch.nn.LeakyReLU())
    backbone.add_module("11", torch.nn.Linear(512, 512))
    backbone.add_module("12", torch.nn.LeakyReLU())
    backbone.add_module("13", torch.nn.Linear(512, 1024))
    backbone.add_module("14", torch.nn.LeakyReLU())
    backbone.add_module("15", torch.nn.Linear(1024, actionsize))

    dqn = DQN(backbone)

    parameters1 = 500, 50, 0.00005, 80, 64
    parameters2 = 500, 50, 0.00005, 50, 64
    parameters3 = 500, 50, 0.00005, 10, 64
    for i in range(25):
        print("================== epoch", i, " ================")
        tuneDQN(dqn, replaybuffer, generator, parameters1)
        tuneDQN(dqn, replaybuffer, generator, parameters2)
        tuneDQN(dqn, replaybuffer, generator, parameters3)

    game = generator.getanother()
    R = 0
    for i in range(50):
        if not game.isvalid:
            break
        print("=====")
        print(game.display(), R)
        R += game.update(dqn.selectsingle(game.getvectorstate()))
    print(game.display(), R)
