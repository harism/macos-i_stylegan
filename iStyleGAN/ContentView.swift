import CoreML
import SwiftUI
import MetalKit
import Accelerate
import AVFoundation

struct ContentView: View {
    @State var image: Image = Image(systemName: "globe")
    @State var renderView: RenderView = RenderView()
    @State var showRenderView: Bool = false
    
    var mappingNetwork: MappingNetwork
    var synthesisNetwork: SynthesisNetwork
    var audioEngine: AVAudioEngine

    init() {
        do {
            let mlModelConfiguration = MLModelConfiguration();
            mlModelConfiguration.computeUnits = MLComputeUnits.all
            mappingNetwork = try MappingNetwork(configuration: mlModelConfiguration)
            synthesisNetwork = try SynthesisNetwork(configuration: mlModelConfiguration)
            audioEngine = AVAudioEngine()
        } catch let error {
            fatalError("\(error)")
        }
    }

    var body: some View {
        VStack {
            ZStack {
                if showRenderView {
                    renderView
                        .frame(width: 2048, height: 1152, alignment: .center)
                } else {
                    image
                        .imageScale(.large)
                        .foregroundColor(.accentColor)
                }
            }.transition(.opacity)
            VStack {
                Text("Hello world!")
                Button("Click me!") {
                    withAnimation {
                        showRenderView = true
                        start()
                    }
                }
            }
            .padding(EdgeInsets(top: 10.0, leading: 10.0, bottom: 10.0, trailing: 10.0))
        }
        .padding(EdgeInsets(top: 0.0, leading: 0.0, bottom: 10.0, trailing: 0.0))
        .preferredColorScheme(.dark)
        .contentMargins(0.0)
    }
}

extension MLMultiArray {
    func interpolate(other: MLMultiArray, iter: Double) {
        do {
            let mlArraySelf = self.dataPointer.bindMemory(to: Float16.self, capacity: 512)
            let mlArrayOther = other.dataPointer.bindMemory(to: Float16.self, capacity: 512)
            let style = try MLMultiArray(shape: [1, 512] as [NSNumber], dataType: MLMultiArrayDataType.float32)
            for index in 0...style.count - 1 {
                var begin = Float(mlArraySelf[index])
                var out = begin.interpolated(towards: Float(mlArrayOther[index]), amount: iter)
                mlArraySelf[index] = Float16(out)
            }
        } catch let error {
            fatalError("\(error)")
        }
    }
    func unsafeRawPointer() -> UnsafeRawPointer {
        return UnsafeRawPointer(self.dataPointer)
    }
    func unsafeMutablePointer() -> UnsafeMutablePointer<Float16> {
        return self.dataPointer.bindMemory(to: Float16.self, capacity: 1024 * 1024 * 3)
    }
    func copy(to: UnsafeMutableRawPointer, size: Int) {
        to.copyMemory(from: self.dataPointer, byteCount: size)
    }
}

private extension ContentView {
    
    func start() {
        do {
            _ = audioEngine.mainMixerNode
            audioEngine.prepare()
            try audioEngine.start()
            
            guard let url = Bundle.main.url(forResource: "Virgill - Just chip it", withExtension: "mp3") else {
                print("mp3 not found")
                return
            }
            
            let player = AVAudioPlayerNode()
            let audioFile = try AVAudioFile(forReading: url)
            let format = audioFile.processingFormat
            audioEngine.attach(player)
            audioEngine.connect(player, to: audioEngine.mainMixerNode, format: format)
            player.scheduleFile(audioFile, at: nil, completionHandler: nil)
            
            var fftSetup = vDSP_DFT_zop_CreateSetup(nil, 1024, vDSP_DFT_Direction.FORWARD)!
            var mappingArray = try MLMultiArray(shape: [1, 512] as [NSNumber], dataType: MLMultiArrayDataType.float16)
            var mappingInput = MappingNetworkInput(var_: mappingArray)
            var mappingOutput : MappingNetworkOutput?
            
            var running : Bool = false
            var rmsBegin : Float = 0.0
            var rmsEnd : Float = 0.0
            var synthesisBegin : MLMultiArray?
            var synthesisEnd : MLMultiArray?
            
            audioEngine.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { (buffer, time) in
                var realIn = [Float](repeating: 0, count: 1024)
                for index in 0...1023 {
                    realIn[index] = buffer.floatChannelData![0][index]
                }
                var imagIn = [Float](repeating: 0, count: 1024)
                var realOut = [Float](repeating: 0, count: 1024)
                var imagOut = [Float](repeating: 0, count: 1024)
                
                var rms : Float = 0
                vDSP_measqv(&realIn, 1, &rms, 1024)
                
                vDSP_DFT_Execute(fftSetup, &realIn, &imagIn, &realOut, &imagOut)
                var complex = DSPSplitComplex(realp: &realOut, imagp: &imagOut)
                var magnitudes = [Float](repeating: 0, count: 512)
                vDSP_zvabs(&complex, 1, &magnitudes, 1, 512)
                
                let timeSec = Double(time.sampleTime) / time.sampleRate

                for index in 0...511 {
                    let idx = Double(index)
                    let mag = Double(magnitudes[index])
                    let val = sin(idx * 0.1 + timeSec * 0.1) + Double(rms) * 0.5 + mag * 0.01
                    mappingArray[index] = val as NSNumber
                }
                
                do {
                    mappingOutput = try mappingNetwork.prediction(input: mappingInput)
                } catch let error {
                    print(error.localizedDescription)
                }
                
                if (!running) {
                    running = true
                    rmsBegin = rms
                    rmsEnd = rms
                    synthesisBegin = mappingOutput?.var_134
                    synthesisEnd = mappingOutput?.var_134
                } else {
                    rmsBegin = rmsEnd
                    rmsEnd = rms
                    synthesisBegin = synthesisEnd
                    synthesisEnd = mappingOutput?.var_134
                    
                    DispatchQueue.global(qos: .userInteractive).async {
                        render(iterations: 5, begin: synthesisBegin!, end: synthesisEnd!, rmsBegin: rmsBegin, rmsEnd: rmsEnd)
                    }
                }
            }
            player.play()
        } catch let error {
            print(error.localizedDescription)
        }
    }

    func render(iterations: Int, begin: MLMultiArray, end: MLMultiArray, rmsBegin: Float, rmsEnd: Float) {
        do {
            for loop in 0..<iterations {
                let iter = Float(loop) / Float(iterations - loop)
                begin.interpolate(other: end, iter: Double(iter))
                let synthesisOutput = try synthesisNetwork.prediction(style: begin)
                let rms = rmsBegin + (rmsEnd - rmsBegin) * (Float(loop) / Float(iterations))
                renderView.render(texture: synthesisOutput.var_1483, rms: rms)
            }
        } catch let error {
            print(error.localizedDescription)
        }
    }
}

struct RenderView : NSViewRepresentable {
    var coordinator : RenderView.Coordinator?
    var view : MTKView = MTKView()

    init() {
        self.coordinator = RenderView.Coordinator(self)
        view.isPaused = true
        view.enableSetNeedsDisplay = true
    }
    
    func makeNSView(context: Context) -> MTKView {
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        return view
    }
    
    func updateNSView(_ uiVIew: MTKView, context : Context) {
    }
    
    func makeCoordinator() -> Coordinator {
        coordinator!
    }
    
    func render(texture: MLMultiArray, rms: Float) {
        coordinator!.update(texture: texture, rms: rms)
        view.draw()
    }
    
    class Coordinator : NSObject, MTKViewDelegate {
        var parent: RenderView
        var metalDevice: MTLDevice!
        var metalCommandQueue: MTLCommandQueue!
        var metalPipelineState: MTLRenderPipelineState!
        var metalVertexBuffer: MTLBuffer!
        var metalTexture: (any MTLTexture)?
        var metalTextureData: UnsafeMutableRawPointer?
        var metalMutex: NSLock = NSLock()
        var metalTextureRms: Float = 0.0
        var metalTextureTime: Float = 0.0
 
        init(_ parent: RenderView) {
            self.parent = parent

            if let metalDevice = MTLCreateSystemDefaultDevice() {
                self.metalDevice = metalDevice
            }
            self.metalCommandQueue = metalDevice.makeCommandQueue()
            
            let metalLibrary = metalDevice.makeDefaultLibrary()
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = metalLibrary?.makeFunction(name: "vertexShader")
            pipelineDescriptor.fragmentFunction = metalLibrary?.makeFunction(name: "fragmentShader")
            pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormat.bgra8Unorm
            do {
                self.metalPipelineState = try metalDevice.makeRenderPipelineState(descriptor: pipelineDescriptor)
            } catch let error {
                print(error.localizedDescription)
            }
            
            let vertices: [Float] = [-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0]
            self.metalVertexBuffer = metalDevice.makeBuffer(bytes: vertices, length: 8 * MemoryLayout<Float>.size, options: [])
            
            let textureDescriptor = MTLTextureDescriptor()
            textureDescriptor.textureType = MTLTextureType.type2D
            textureDescriptor.usage = MTLTextureUsage.shaderRead
            textureDescriptor.storageMode = MTLStorageMode.shared
            textureDescriptor.pixelFormat = MTLPixelFormat.r16Float
            textureDescriptor.width = 1024 * 3
            textureDescriptor.height = 1024
            self.metalTexture = metalDevice.makeTexture(descriptor: textureDescriptor)
            
            self.metalTextureData = UnsafeMutableRawPointer.allocate(byteCount: 1024 * 1024 * 3 * MemoryLayout<Float16>.stride, alignment: 1)
            
            super.init()
        }
        
        func update(texture: MLMultiArray, rms: Float) {
            metalMutex.lock()
            metalTextureRms = rms
            metalTextureTime += rms + 0.005
            texture.copy(to: metalTextureData!, size: 1024 * 1024 * 3 * MemoryLayout<Float16>.stride)
            metalMutex.unlock()
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        }
        
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable else {
                return
            }
            
            let commandBuffer = metalCommandQueue.makeCommandBuffer()

            let renderPassDescriptor = view.currentRenderPassDescriptor
            renderPassDescriptor!.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 1, 1)
            renderPassDescriptor!.colorAttachments[0].loadAction = .clear
            renderPassDescriptor!.colorAttachments[0].storeAction = .store
            
            metalMutex.lock()
            let region = MTLRegionMake2D(0, 0, 1024 * 3, 1024)
            metalTexture?.replace(region: region, mipmapLevel: 0, withBytes: metalTextureData!, bytesPerRow: 3 * 1024 * MemoryLayout<Float16>.stride)
            
            let renderCommandEncoder = commandBuffer!.makeRenderCommandEncoder(descriptor: renderPassDescriptor!)
            renderCommandEncoder!.setRenderPipelineState(metalPipelineState)
            renderCommandEncoder!.setVertexBuffer(metalVertexBuffer, offset: 0, index: 0)
            renderCommandEncoder!.setFragmentBytes(&metalTextureRms, length: MemoryLayout.size(ofValue: metalTextureRms), index: 0)
            renderCommandEncoder!.setFragmentBytes(&metalTextureTime, length: MemoryLayout.size(ofValue: metalTextureTime), index: 1)
            renderCommandEncoder!.setFragmentTexture(metalTexture, index: 0)
            renderCommandEncoder!.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderCommandEncoder!.endEncoding()
            
            commandBuffer!.present(drawable)
            commandBuffer!.commit()
            metalMutex.unlock()
        }
    }
}
