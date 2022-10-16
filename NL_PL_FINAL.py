import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
import scipy as sc
import sys


class NODE:
    def __init__(self,id,x,y):
        self.id=id
        self.x=x
        self.y=y
        self.dof=[0,0]
        
class ELEMENT:
    def __init__(self,id,n1,n2):
        self.id=id
        self.nodeList=[n1,n2]
        self.area=1
        self.N=0
        self.exten=0
        self.E0=0
        self.yieldstress=210000000*0.00654
        self.stress=0
        self.alpha=0
    def lengh(self):
        self.dx=self.nodeList[1].x-self.nodeList[0].x
        self.dy=self.nodeList[1].y-self.nodeList[0].y
        self.L=(self.dx**2+self.dy**2)**(0.5)
    def B(self):
        self.lengh()
        self.cos=self.dx/self.L
        self.sen=self.dy/self.L
        self.b=(1/self.L)*np.array([self.cos,self.sen,-(self.cos),-(self.sen)])
    def stiff(self,E0):
        self.lengh()
        self.B()
        k=(E0*self.L)*(self.b.reshape((4,1))*self.b.reshape((1,4)))
        return k 
        
class LOAD:
    def __init__(self,Fx,Fy,n):
        self.Fx=Fx
        self.Fy=Fy
        self.node=n
        
class TRUSS:
    def __init__(self):
        self.nodeList=[]
        self.elemList=[]
        self.loadList=[]
        self.addLoadList=[]
    
    def addNode(self,n): self.nodeList.append(n)
    def addElem(self,e): self.elemList.append(e)    
    def addLoad(self,l): self.loadList.append(l) 
    
    def nrnodes(self): return len(self.nodeList)
    def nrelems(self): return len(self.elemList)
    def nrloads(self): return len(self.loadList)
    
    def numberingDOF(self):
        count=0
        for node in self.nodeList:
            for i in [0,1]:
                if node.dof[i]==0 : 
                    count+=1
                    node.dof[i]=count
        return count
    
    def computeLoads(self):
        F=[0]*self.NDOF
        for load in self.loadList:
            if load.node.dof[0]!=-1: F[load.node.dof[0]-1]=load.Fx
            if load.node.dof[1]!=-1: F[load.node.dof[1]-1]=load.Fy
        self.Fglobal=F
        #print("Global Force Vector", self.Fglobal)

    def LoadMultiplier(self,a):
        self.LM=a
        #print("Load Multiplier",self.LM)

    def stress(self,eps0,E0):
        for elems in self.elemList:
            e=elems.exten
            a=eps0
            b=E0
            if e<=a and e>=-a :
                elems.stress=e*b 
            if e>a :
                c=-(a*b-(9./10.)*b*(a-a*np.log(abs(a))))+a*b
                elems.stress=e*b-(9./10.)*b*(e-a*np.log(abs(e)))+c
            if e<-a :
                c=-(-a*b-(9./10.)*b*(-a+a*np.log(abs(a))))-a*b 
                elems.stress=e*b-(9./10.)*b*(e+a*np.log(abs(e)))+c 
            #print("Element Normal Stress - (Function)","e#",elems.id,elems.stress)
    
    def stresspl(self,eps0,E0,H):
        for elems in self.elemList:
            a=1-self.PF
            b=E0
            trialstress=elems.stress+b*elems.addexten
            if  abs(trialstress-elems.alpha) <= elems.yieldstress:
                elems.stress=trialstress
            else:
                epsplc=a*(abs(trialstress-elems.alpha)-elems.yieldstress)/(b+H)
                epspli=self.PF*(abs(trialstress)-elems.yieldstress)/(b+H)
                if trialstress-elems.alpha>0: sign=1
                else: sign=-1
                if trialstress>0: sign1=1
                else: sign1=-1
                elems.stress=trialstress-b*(epsplc*sign+epspli*sign1)
                elems.alpha+=H*epsplc*sign
                elems.yieldstress+=H*epspli

        #print("Element Alpha Value -",elems.id,"=",elems.alpha)
        #print("Element Yield Stress -",elems.id,"=",elems.yieldstress)
        #print("Element Trial Stress -",elems.id,"=",trialstress)
        #print("Element ExtensionPL -",elems.id,"=",elems.epspl)
        #print("Element StressPL -",elems.id,"=",elems.stress)
        #print("Element Corrected Alpha Value -",elems.id,"=",elems.alpha)
        #print("Element Crrected Yield Stress -",elems.id,"=",elems.yieldstress)
                
    def N(self):
        for elems in self.elemList:
            elems.N=elems.stress*elems.area
            #print("Element Normal Force - (Function)","e#",elems.id,elems.N)

    def FLOCAL(self):
        for elems in self.elemList:
            elems.lengh()
            elems.B()
            elems.FLOCAL=elems.L*np.transpose(elems.b)*elems.N
            #print("Local Force Vector - (Function)","e#",elems.id,elems.FLOCAL)

    def computeQ(self):
        self.Q=[0]*self.NDOF
        for elems in self.elemList:
            for i in [-1,0]:
                for j in [0,1]:
                    if i == -1 and j==0 : a=0
                    if i == -1 and j==1 : a=1
                    if i == 0 and j==0 : a=2
                    if i == 0 and j==1 : a=3
                    if elems.nodeList[i].dof[j]!=-1 : 
                        self.Q[elems.nodeList[i].dof[j]-1]+=elems.FLOCAL[a]
                        #Same of: Qu[elems.nodeList[1].dof[0]-1]=Q[elems.nodeList[1].dof[0]-1]+a
        return np.array(self.Q)

    def globalstiff(self,E0):
        rows=[]
        cols=[]
        vals=[]
        for elem in self.elemList:
            for a in [-1,0]:
                for b in [0,1]:
                    for c in [-1,0]:
                        for d in [0,1]:
                            if elem.nodeList[a].dof[b]!=-1 and elem.nodeList[c].dof[d]!=-1 : 
                                rows.append(elem.nodeList[a].dof[b]-1)
                                cols.append(elem.nodeList[c].dof[d]-1)
                                if c==-1 and d==0 : i=0
                                if c==-1 and d==1 : i=1
                                if c==0 and d==0 : i=2
                                if c==0 and d==1 : i=3
                                if a==-1 and b==0 : j=0
                                if a==-1 and b==1 : j=1
                                if a==0 and b==0 : j=2
                                if a==0 and b==1 : j=3
                                vals.append(elem.stiff(E0)[i,j])
        Kglobal=coo_matrix((vals, (rows, cols)), shape=(self.NDOF,self.NDOF))
        self.Kglobal=splu(Kglobal.tocsc())

    def extension(self):
        for elems in self.elemList:
            if elems.nodeList[-1].dof[0]==-1 : a=0 
            else: a=self.d[(elems.nodeList[-1].dof[0]-1)]
            if elems.nodeList[-1].dof[1]==-1 : b=0
            else: b=self.d[(elems.nodeList[-1].dof[1]-1)]
            if elems.nodeList[0].dof[0]==-1 : c=0
            else: c=self.d[(elems.nodeList[0].dof[0]-1)]
            if elems.nodeList[0].dof[1]==-1 : d=0
            else: d=self.d[(elems.nodeList[0].dof[1]-1)]
            elems.displelem2=np.array([a,b,c,d])
            elems.addexten=elems.b.dot(elems.displelem2)
            elems.exten+=elems.addexten
            #print("Element Additional Extension-","Element number-",elems.id,"=",elems.addexten)
            #print("Element Total Extension-","Element number-",elems.id,"=",elems.exten)
        
    def results(self,E0):
        self.FLOCAL()
        self.computeQ() 
        #print("Vector Q",self.Q)
        self.computeLoads()
    
    def line(self):
        print("-" * 100)
    
    def mark(self):
        print("~" * 100)
        
    def RUN(self,eps0,E0,PROBLEMTYPE,H,PF):
        self.mark()
        if PROBLEMTYPE==1: print("NON LINEAR ELASTICITY")
        if PF==0: W="KINEMATIC HARDENING"
        if PF==1: W="ISOTROPIC HARDENING"
        if PF!=0 and PF!=1 : W="COMBINED ISOTROPIC-KINEMATIC"
        if PROBLEMTYPE==2: print(W,"PLASTICITY")
        print("LOAD PATH",self.LM)
        self.mark()
        self.NDOF=self.numberingDOF()
        self.globalstiff(E0)
        self.TD=[0]*self.NDOF
        LOADPATH_ORDER=0
        self.PT=PROBLEMTYPE
        self.PF=PF
        self.history=[(0,[0]*self.NDOF,[elems.exten for elems in self.elemList],[elems.N for elems in self.elemList],[elems.stress for elems in self.elemList])]
        for l in self.LM:
            count=0
            LOADPATH_ORDER+=1
            self.results(E0)
            self.T=[]
            for i in range(0,len(self.Fglobal)): self.T.append(self.Fglobal[i]*l)
            self.DELTAF=self.T-self.computeQ()
            while self.DELTAF.dot(self.DELTAF)**.5>=0.01 :
                count+=1
                self.d=self.Kglobal.solve(np.array(self.DELTAF))
                self.TD+=self.d
                self.extension()
                if self.PT==1 : self.stress(eps0,E0)
                if self.PT==2 : self.stresspl(eps0,E0,H)
                self.N()
                self.results(E0)
                self.DELTAF=self.T-self.computeQ()
            self.history.append((l,self.TD.tolist(),[elems.exten for elems in self.elemList],[elems.N for elems in self.elemList],[elems.stress for elems in self.elemList]))
            print("LOAD PATH ORDER=",LOADPATH_ORDER,"LOAD MULTIPLIER=",l)
            print("# of iterations=",count)              
            print("DELTAF=",self.DELTAF)
            print("Global Force Vector=",self.T)
            print("Total displacements=",self.TD)
            for elems in self.elemList:
                print("Element Normal Force-(Given by Function)","element number-",elems.id,"=",elems.N)
                print("Element Additional Extension-","Element number-",elems.id,"=",elems.addexten)
                print("Element Total Extension-","Element number-",elems.id,"=",elems.exten)
                print("Element Normal Stress -","Element number-",elems.id,"=",elems.stress)
            self.line()
            self.line()
    
    def show(self):
        ax = plt.axes()

        for i,e in enumerate(self.elemList):
            plt.plot([e.nodeList[0].x,e.nodeList[1].x], [e.nodeList[0].y,e.nodeList[1].y], color= 'b')
            x=(e.nodeList[0].x+e.nodeList[1].x)/2
            y=(e.nodeList[0].y+e.nodeList[1].y)/2

        for n in self.nodeList:
            if n.dof[0]>-1 and n.dof[1]>-1:
                plt.scatter(n.x, n.y, marker='o', color= 'b')
            elif n.dof==[-1,-1]:
                plt.scatter(n.x, n.y, s=100, marker='s', color= 'r')
            elif n.dof[1]==-1:
                plt.scatter(n.x, n.y, s=100, marker='^', color= 'r')
            elif n.dof[0]==-1:
                plt.scatter(n.x, n.y, s=100, marker='>', color= 'r')

        for l in self.loadList:
            plt.quiver(l.node.x, l.node.y, l.Fx, l.Fy, scale=3000000, color="g")

        min, max=ax.get_ylim()
        tol=(max-min)*.2
        ax.set_ylim((min-tol,max+tol))
        min, max=ax.get_xlim()
        tol=(max-min)*.2
        ax.set_xlim((min-tol,max+tol))

        ax.axis('off')
        plt.show()
 
#INSERÇÃO DOS DADOS
#NÓS
n1=NODE(1,0,0)
n2=NODE(2,1,0)
n3=NODE(3,2,1)
#FRONTEIRAS
n1.dof=[-1,-1]  
n2.dof=[0,-1]
n3.dof=[-1,-1]
#ELEMENTOS
e1=ELEMENT(1,n1,n2)
e2=ELEMENT(2,n2,n3)
#FORÇAS
f1=LOAD(457800,0,n2)
#f1=LOAD(-915600,0,n3)
#EXEMPLO
exemplo=TRUSS()
exemplo.addNode(n1)
exemplo.addNode(n2)
exemplo.addNode(n3)
exemplo.addElem(e1)
exemplo.addElem(e2)
exemplo.addLoad(f1)
#CAMINHO DE CARGA
exemplo.LoadMultiplier(a=[1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0])

#LOAD PATH UTILIZADOS
#a=[1,2,3,4,5,6,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-5,-4,-3,-2,-1,0] - LOAD MATH - KINEMATIC HARDENING
#a=[1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0] - LOAD PATH - ISOTROPIC HARDENING AND NON-LINEAR
#eps UTILIZADOS (TROCAR TANTO NA INSERÇÃO DOS DADOS QUANTO NA CLASSE DO ELEMENTO)
#eps0=0.00654 - PLASTICITY
#eps0=0.00436 - NON-LINEAR ELASTICITY

#INSERÇÃO DOS COMANDOS DE RESPOSTA:
    #PROBLEMTYPE=1 - NON LINEAR PROBLEM
    #PROBLEMTYPE=2 - PLASTICITY - PF=0 - KINEMATIC HARDENING // PF=1 - ISOTROPIC HARDENING

exemplo.RUN(eps0=0.00654,E0=210000000,PROBLEMTYPE=2,H=210000000,PF=1)

#GRÁFICOS

exemplo.show()

plt.plot([h[1][0] for h in exemplo.history],[h[0] for h in exemplo.history], marker=".")
plt.xlabel("displacement")
plt.ylabel("load multiplier")
plt.grid(True)
plt.show()

plt.plot([h[2][0] for h in exemplo.history],[h[3][0] for h in exemplo.history], marker=".")
plt.xlabel("strains")
plt.ylabel("N")
plt.grid(True)
plt.show()

plt.plot([h[2][0] for h in exemplo.history],[h[-1][0] for h in exemplo.history], marker=".")
plt.xlabel("strains")
plt.ylabel("Stress")
plt.grid(True)
plt.show()

