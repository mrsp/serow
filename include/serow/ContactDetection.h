#include <serow/Gaussian.h>
#include <string.h>




namespace serow
{
class ContactDetection
{
private:
  Gaussian gs;
  //Probabilistic Kinematic - Contact Wrench detection
  double sigmalf, sigmarf;
  double sigmalc, sigmarc;
  double sigmalv, sigmarv;
  double xmin, xmax, ymin, ymax;
  double lfmin, rfmin;
  double plf, prf, plc, prc, plv, prv;
  double pl, pr, p;
  double VelocityThres;
  bool firstContact, useCOP, useKin;
  int contactL, contactR, sum, contactRFilt, contactLFilt;
  double prob_TH;
  std::string support_foot_frame, support_leg, lfoot_frame, rfoot_frame, phase;
  int medianWindow;
	Mediator *lmdf, *rmdf;
  double deltalfz, deltarfz, lf_, rf_;
  //Smidtt Trigger detection
  double LegLowThres, LegHighThres, StrikingContact;

  double computeForceContactProb(double fmin, double sigma, double f)
  {
    return 1.000 - gs.cdf(fmin, f, sigma);
  }

  double computeKinContactProb(double vel_min, double sigma, double v)
  {
    return gs.cdf(vel_min, v, sigma);
  }


  double computeCOPContactProb(double max, double min, double sigma, double cop)
  {
    if (cop != 0)
    {
      return gs.cdf(max, cop, sigma) - gs.cdf(min, cop, sigma);
    }
    else
    {
      return 0;
    }
  }

  double computeContactProb(double lf, double rf, double coplx, double coply, double coprx, double copry, double lv, double rv)
  {

    plf = computeForceContactProb(lfmin, sigmalf, lf);
    pl = plf;

    prf = computeForceContactProb(rfmin, sigmarf, rf);
    pr = prf;

    //To utilize the full contact-wrench - careful for non-coplanar contacts
    if (useCOP)
    {
      plc = computeCOPContactProb(xmax, xmin, sigmalc, coplx) * computeCOPContactProb(ymax, ymin, sigmalc, coply);
      prc = computeCOPContactProb(xmax, xmin, sigmarc, coprx) * computeCOPContactProb(ymax, ymin, sigmarc, copry);
      pl *= plc;
      pr *= prc;
    }
    if (useKin)
    {
      plv = computeKinContactProb(VelocityThres, sigmalv, lv);
      prv = computeKinContactProb(VelocityThres, sigmarv, rv);
      pl *= plv;
      pr *= prv;
    }

    p = pl + pr;

    if (p != 0)
    {
      pl = pl / p;
      pr = pr / p;
    }
    else
    {
      pl = 0;
      pr = 0;
    }

    contactL = 0;
    if (pl >= prob_TH)
      contactL = 1;

    contactR = 0;
    if (pr >= prob_TH)
      contactR = 1;

    MediatorInsert(lmdf, contactL);
    contactLFilt = MediatorMedian(lmdf); 
    MediatorInsert(rmdf, contactR);
    contactRFilt = MediatorMedian(rmdf);


    sum = contactRFilt + contactLFilt;

    switch(sum)
    {
    case 0:
      phase = "None";
      break;
    case 1:
      phase = "Single Support";
      break;
    case 2:
      phase = "Double Support";
      break;
    }

  }

public:
  void init(std::string lfoot_frame_, std::string rfoot_frame_, double lfmin_, double rfmin_, double xmin_, double xmax_,
            double ymin_, double ymax_, double sigmalf_, double sigmarf_, double sigmalc_, double sigmarc_, double VelocityThres_, double sigmalv_, double sigmarv_, bool useCOP_ = false, bool useKin_ = false, double prob_TH_ = 0.9 , int medianWindow_=  10)
  {
    lfoot_frame = lfoot_frame_;
    rfoot_frame = rfoot_frame_;
    lfmin = lfmin_;
    rfmin = rfmin_;
    xmin = xmin_;
    xmax = xmax_;
    ymin = ymin_;
    ymax = ymax_;
    sigmalf = sigmalf_;
    sigmarf = sigmarf_;
    sigmalc = sigmalc_;
    sigmarc = sigmarc_;
    sigmalv = sigmalv_;
    sigmarv = sigmarv_;
    contactL = 0;
    contactR = 0;
    contactRFilt = 0;
    contactLFilt = 0;
    pl = 0;
    pr = 0;
    plf = 0;
    prf = 0;
    plc = 0;
    prc = 0;
    plv = 0;
    prv = 0;
    VelocityThres = VelocityThres_;
    prob_TH = prob_TH_/2.0;
    useCOP = useCOP_;
    useKin = useKin_;
    firstContact = true;
    deltalfz = 0;
    deltarfz = 0;
    medianWindow = medianWindow_;
    rmdf = MediatorNew(medianWindow);
    lmdf = MediatorNew(medianWindow);

    cout<<"Gait-Phase Estimation Module Initialized"<<endl;
  }

  void init(std::string lfoot_frame_, std::string rfoot_frame_, double LegHighThres_, double LegLowThres_, double StrikingContact_, double VelocityThres_, double prob_TH_ = 0.9, int medianWindow_=  10)
  {

    lfoot_frame = lfoot_frame_;
    rfoot_frame = rfoot_frame_;
    LegHighThres = LegHighThres_;
    LegLowThres = LegLowThres_;
    StrikingContact = StrikingContact_;
    VelocityThres = VelocityThres_;
    firstContact = true;
    prob_TH = prob_TH_;
    pl = 0;
    pr = 0;
    deltalfz = 0;
    deltarfz = 0;
    contactL = 0;
    contactR = 0;
    contactRFilt = 0;
    contactLFilt = 0;
    medianWindow = medianWindow_;
    rmdf = MediatorNew(medianWindow);
    lmdf = MediatorNew(medianWindow);
    cout<<"Gait-Phase Estimation Module with Schmitt-Trigger Initialized"<<endl;
  }
  double getDiffForce()
  {
    cout<<"Delta FORCE"<<endl;
    cout<<fabs(deltalfz + deltarfz)<<endl;
    return fabs(deltalfz + deltarfz);
  }
  void computeSupportFoot(double lf, double rf, double coplx, double coply, double coprx, double copry, double lvnorm, double rvnorm)
  {

   
    computeContactProb(lf, rf, coplx, coply, coprx, copry, lvnorm, rvnorm);
    //Choose Initial Support Foot based on Contact Force
      if (contactLFilt ==1 && contactRFilt == 0)
      {
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else if (contactRFilt ==1 && contactLFilt == 0)
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
      else if (contactRFilt == 1 && contactLFilt == 1)
      {
        if(pl>pr)
        {
          support_leg = "LLeg";
          support_foot_frame = lfoot_frame;
        }
        else
        {
         support_leg = "RLeg";
         support_foot_frame = rfoot_frame;
        }
        
      }
      else
      {
        support_leg = "None";
         support_foot_frame = "None";
      }
    

    if(!firstContact)
    {
      deltalfz = lf - lf_;
      deltarfz = rf - rf_;
    }
    else
    {
      firstContact = false; 
    }
    
    lf_ = lf;
    rf_ = rf;
  }

  double getLLegContactProb()
  {
    return pl;
  }

  double getRLegContactProb()
  {
    return pr;
  }

  int isLLegContact()
  {
    return contactLFilt;
  }

  int isRLegContact()
  {
    return contactRFilt;
  }

  std::string getSupportFrame()
  {
    return support_foot_frame;
  }
  std::string getSupportLeg()
  {
    return support_leg;
  }
  std::string getSupportPhase()
  {
    return phase;
  }

  void computeForceWeights(double lf, double rf)
  {
    p = lf + rf;
    pl = 0;
    pr = 0;

    if (p > 0)
    {
      pl = lf / p;
      pr = rf / p;
    }
    if(pl < 0)
      pl = 0;
    
    if(pl > 1.0)
      pl = 1.0;

    if(pr < 0)
      pr = 0;

    if(pr > 1.0)
      pr = 1.0;



  }
  void SchmittTrigger(double lf, double rf)
  {
  



    if (contactL == 0)
    {
      if (lf > LegHighThres )
      {
        contactL = 1;
      }
    }
    else
    {
      if (lf < LegLowThres)
      {
        contactL = 0;
      }
    }

    if (contactR == 0)
    {
      if (rf > LegHighThres)
      {
        contactR = 1;
      }
    }
    else
    {
      if (rf < LegLowThres)
      {
        contactR = 0;
      }
    }



    MediatorInsert(lmdf, contactL);
    contactLFilt = MediatorMedian(lmdf); 
    MediatorInsert(rmdf, contactR);
    contactRFilt = MediatorMedian(rmdf);

  

    


    sum = contactRFilt + contactLFilt;

    switch(sum)
    {
    case 0:
      phase = "None";
      break;
    case 1:
      phase = "Single Support";
      break;
    case 2:
      phase = "Double Support";
      break;
    }

    if (contactLFilt == 1 && contactRFilt == 1)
    {
      if(pl>pr){
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
      
    }
    else if(contactLFilt == 1)
    {
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
    }
    else if(contactRFilt == 1)
    {
      support_leg = "RLeg";
      support_foot_frame = rfoot_frame;
    }
    else
    {
      support_leg = "None";
      support_foot_frame = "None";
    }
    
    if (!firstContact)
    {
      deltalfz = lf - lf_;
      deltarfz = rf - rf_;
    }
    else
    {
      firstContact = false;
    }

    lf_ = lf;
    rf_ = rf;
  }
 };
}