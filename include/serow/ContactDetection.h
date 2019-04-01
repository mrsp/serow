#include <serow/Gaussian.h>
#include <string.h>
#include <eigen3/Eigen/Dense>


namespace serow{
class ContactDetection
{
private:
  Gaussian gs;
  double sigmalf, sigmarf;
  double sigmalc, sigmarc;
  double xmin, xmax, ymin, ymax;
  double lfmin, rfmin;
  double plf, prf, plc, prc;
  double pl, pr, p;
  double VelocityThres;
  std::string support_foot_frame, support_leg, lfoot_frame, rfoot_frame;
  
  double computeForceContactProb(double fmin, double sigma, double f)
  {
    return 1.000 - gs.cdf(fmin, f, sigma);
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

  double computeContactProb(double lf, double rf, double coplx, double coply, double coprx, double copry)
  {

    plf = computeForceContactProb(lfmin, sigmalf, lf);
    prf = computeForceContactProb(rfmin, sigmarf, rf);
    plc = computeCOPContactProb(xmax, xmin, sigmalc, coplx) * computeCOPContactProb(ymax, ymin, sigmalc, coply);
    prc = computeCOPContactProb(xmax, xmin, sigmarc, coprx) * computeCOPContactProb(ymax, ymin, sigmarc, copry);
    pr = prf * prc;
    pl = plf * plc;
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
  }

public:
  bool firstContact;
  void init(std::string lfoot_frame_, std::string rfoot_frame_, double lfmin_, double rfmin_, double xmin_, double xmax_,
            double ymin_, double ymax_, double sigmalf_, double sigmarf_, double sigmalc_, double sigmarc_, double VelocityThres_)
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
    VelocityThres = VelocityThres_;
    firstContact = true;
  }

  double getLLegContactProb()
  {
    return pl;
  }

  double getRLegContactProb()
  {
    return pr;
  }
  std::string getSupport()
  {
    return support_foot_frame;
  }
  void computeSupportFoot(double lf, double rf, double coplx, double coply, double coprx, double copry, Eigen::Vector3d vwl, Eigen::Vector3d vwr)
  {

    computeContactProb( lf,  rf,  coplx,  coply,  coprx,  copry);
    //Choose Initial Support Foot based on Contact Force
    if (firstContact)
    {
      if (pl > pr)
      {
        // Initial support leg
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
      firstContact = false;
    }
    else
    {
      if ((pl >= pr) && (vwl.norm() < VelocityThres && vwr.norm() > VelocityThres))
      {
        support_leg = "LLeg";
        support_foot_frame = lfoot_frame;
      }
      else if ((pr >= pl) && (vwr.norm() < VelocityThres && vwl.norm() > VelocityThres))
      {
        support_leg = "RLeg";
        support_foot_frame = rfoot_frame;
      }
    }
  }
};
}
