#include "landmarker.h"
#include "pfldlandmarker/pfldlandmarker.h"
#include "zqlandmarker/zqlandmarker.h"
#include "Peppa_Pig_Face_Landmark/Peppa_Pig_Face_Landmark.h"

namespace mirror {
Landmarker* PFLDLandmarkerFactory::CreateLandmarker() {
    return new PFLDLandmarker();
}

Landmarker* ZQLandmarkerFactory::CreateLandmarker() {
    return new ZQLandmarker();
}

Landmarker* Peppa_PigLandmarkerFactory::CreateLandmarker() {
    return new Peppa_Pig_Face_Landmarker();
}

}