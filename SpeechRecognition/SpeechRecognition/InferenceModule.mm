// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#import "InferenceModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>
#import <AVFoundation/AVAudioRecorder.h>
#import <AVFoundation/AVAudioSettings.h>
#import <AVFoundation/AVAudioSession.h>
#import <AVFoundation/AVAudioPlayer.h>


@implementation InferenceModule {
    
    @protected torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            auto qengines = at::globalContext().supportedQEngines();
            if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
                at::globalContext().setQEngine(at::QEngine::QNNPACK);
            }
            _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
        }
        catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}


- (NSString*)recognize:(void*)wavBuffer bufLength:(int)bufLength{
    try {
        at::Tensor tensorInputs = torch::from_blob((void*)wavBuffer, {1, bufLength}, at::kFloat);
        CFTimeInterval startTime = CACurrentMediaTime();
        auto result = _impl.forward({ tensorInputs });
        CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        NSLog(@"inference time:%f", elapsedTime);
            
        return [NSString stringWithCString:result.toStringRef().c_str() encoding:[NSString defaultCStringEncoding]];
    }
    catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

- (NSArray<NSNumber*>*)vocode:(void*)units lang:(NSString*)lang{
    try {
        at::Tensor tensorInputs = torch::tensor({991,331,333,873,32,683,589,26,204,280,534,485,974,813,627,545,711,510,884,79,868,220,498,172,871,822,89,664,990,107,522,519,26,204,280,576,384,879,443,93,545,85,510,337,243,850,260,978,56,165,319,501,137,366,641,347,124,362,493,361,931,878,538,423,663,969,70,918,743,955,333,437,85,337,243,889,324,826,789,677,253,355,692,747,671,877,488,443,93,274,208,944,955,865,641,124,243,116,475,783,104,430,945,29,759,973,288,796,33,432,742,924,866,261,230,976,534,485,321,948,885,555,233,156,824,556,655,837,81,194,664,506,686,613,417,755,237,193,415,772}, {at::kInt});
        // TODO: Add lang argument
        std::unordered_map<std::string, at::Tensor> codeInputs = {{ "code", tensorInputs }};
        auto tensorOutputs = _impl.forward({ codeInputs }).toTensor();
        float* floatBuffer = tensorOutputs.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        NSMutableArray* result = [[NSMutableArray alloc] init];
        for (int i = 0; i < tensorOutputs.numel(); i++) {
            [result addObject:@(floatBuffer[i])];
        }
        return [result copy];
    }
    catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}


@end
